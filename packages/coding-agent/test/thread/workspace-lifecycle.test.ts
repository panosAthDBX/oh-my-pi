import { expect, it, spyOn } from "bun:test";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { MessageChannel } from "node:worker_threads";
import { TempDir } from "@oh-my-pi/pi-utils";
import { getWorkspaceLifecycleService, WORKSPACE_LEASE_FILE } from "../../src/task/workspace-lifecycle";
import type { MainThreadLink } from "../../src/thread/protocol";
import { type RpcFrame, RpcLink } from "../../src/thread/rpc";
import type { WorkerPeer } from "../../src/thread/services";
import { workspaceLifecycleThreadService } from "../../src/thread/services/workspace-lifecycle";
import type { WorkspaceProbe, WorkspaceTestGlobals } from "../fixtures/workspace-lifecycle-worker";

it("main gates and inventories worker-owned leases, refuses stale generations, and holds a dead worker's workspace", async () => {
	using temp = TempDir.createSync("omp-workspace-thread-");
	const worker = new Worker(new URL("../fixtures/workspace-lifecycle-worker.ts", import.meta.url).href, { type: "module" });
	const channel = new MessageChannel();
	const counter = new SharedArrayBuffer(4);
	const ready = Promise.withResolvers<void>();
	const onReady = (event: MessageEvent): void => {
		if (event.data?.ready) {
			worker.removeEventListener("message", onReady);
			ready.resolve();
		}
	};
	worker.addEventListener("message", onReady);
	worker.postMessage({ sync: channel.port2, counter }, [channel.port2]);
	await ready.promise;
	const link: MainThreadLink = new RpcLink({
		send: frame => worker.postMessage(frame),
		listen: handler => {
			const listener = (event: MessageEvent<RpcFrame>): void => handler(event.data);
			worker.addEventListener("message", listener);
			return () => worker.removeEventListener("message", listener);
		},
	}, { name: "main-workspace-fixture", syncReplies: { port: channel.port1, counter: new Int32Array(counter) } });
	let holds = 0;
	const peer: WorkerPeer = { threadId: "w1", link, retain: () => { holds++; return () => { holds--; }; } };
	workspaceLifecycleThreadService.installMain?.(peer);
	link.handle("ping", nonce => nonce);
	const connected = Promise.withResolvers<WorkspaceProbe>();
	const test: WorkspaceTestGlobals = { ready: probe => connected.resolve(probe) };
	const service = getWorkspaceLifecycleService();
	const id = `WorkerLease-${path.basename(temp.path())}`;
	const seen: string[] = [];
	const stop = service.registerReleaseGate(async lease => {
		seen.push(lease.agentId);
		return { ok: false, hold: "main plugin preservation hold" };
	});
	try {
		await link.call("boot", { [workspaceLifecycleThreadService.name]: workspaceLifecycleThreadService.globals?.(peer), test });
		const probe = await connected.promise;
		const init = {
			agentId: id, keepAlive: true, repoRoot: temp.path(), workspaceDir: temp.path(), baseDir: temp.path(), backend: "fixture",
			baseline: { commit: "base", source: "working-tree" as const, nested: "source-working-copy" as const },
		};
		const first = await probe.register(init);
		expect(first.threadId).toBeGreaterThan(0);
		expect(holds).toBe(1);
		expect((await probe.list()).find(lease => lease.agentId === id)).toEqual(service.list().find(lease => lease.agentId === id));
		expect(await service.release(id, { generation: first.generation, reason: "busy" })).toMatchObject({ status: "refused", reason: "active-owner: capture in progress" });
		await probe.finish(id, first.generation);
		expect(await service.release(id, { generation: "stale", reason: "stale" })).toMatchObject({ status: "refused", reason: "stale-generation" });
		expect(await service.release(id, { generation: first.generation, reason: "preserve" })).toMatchObject({ status: "held", reason: "main plugin preservation hold" });
		expect(seen).toEqual([id]);
		expect(holds).toBe(0);
		expect(await Bun.file(path.join(temp.path(), WORKSPACE_LEASE_FILE)).json()).toMatchObject({ generation: first.generation, state: "held" });
		const second = await probe.register(init);
		expect(await service.release(id, { generation: first.generation, reason: "old" })).toMatchObject({ status: "refused", reason: "stale-generation" });
		const persisted = Promise.withResolvers<void>();
		const holdAtShutdown = service.holdAtShutdown.bind(service);
		// Observe the actual atomic sidecar completion, not a timer or filesystem poll.
		const hold = spyOn(service, "holdAtShutdown").mockImplementation(async (...args) => {
			await holdAtShutdown(...args);
			persisted.resolve();
		});
		try {
			link.close("worker crashed");
			await worker.terminate();
			workspaceLifecycleThreadService.onWorkerExit?.(peer, "test crash");
			await persisted.promise;
			expect(await Bun.file(path.join(temp.path(), WORKSPACE_LEASE_FILE)).json()).toMatchObject({ generation: second.generation, state: "held", holdReason: "workspace owner worker exited: test crash" });
		} finally { hold.mockRestore(); }
	} finally {
		stop();
		link.close("done");
		await worker.terminate();
		workspaceLifecycleThreadService.onWorkerExit?.(peer, "done");
		await fs.rm(path.join(temp.path(), WORKSPACE_LEASE_FILE), { force: true });
	}
}, 30_000);
