import { expect, it, spyOn } from "bun:test";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { MessageChannel } from "node:worker_threads";
import { TempDir } from "@oh-my-pi/pi-utils";
import {
	getWorkspaceLifecycleHost,
	getWorkspaceLifecycleService,
	WORKSPACE_LEASE_FILE,
	type WorkspaceLease,
	type WorkspaceReleaseGate,
} from "../../src/task/workspace-lifecycle";
import type { MainThreadLink } from "../../src/thread/protocol";
import { type RpcFrame, RpcLink } from "../../src/thread/rpc";
import type { WorkerPeer } from "../../src/thread/services";
import { workspaceLifecycleThreadService } from "../../src/thread/services/workspace-lifecycle";
import type { WorkspaceProbe, WorkspaceTestGlobals } from "../fixtures/workspace-lifecycle-worker";

/** A real Bun worker booted with the workspace lifecycle thread service, linked to this (main) thread. */
async function startWorker(threadId: string) {
	const worker = new Worker(new URL("../fixtures/workspace-lifecycle-worker.ts", import.meta.url).href, {
		type: "module",
	});
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
	const link: MainThreadLink = new RpcLink(
		{
			send: frame => worker.postMessage(frame),
			listen: handler => {
				const listener = (event: MessageEvent<RpcFrame>): void => handler(event.data);
				worker.addEventListener("message", listener);
				return () => worker.removeEventListener("message", listener);
			},
		},
		{ name: "main-workspace-fixture", syncReplies: { port: channel.port1, counter: new Int32Array(counter) } },
	);
	let holds = 0;
	const peer: WorkerPeer = {
		threadId,
		link,
		retain: () => {
			holds++;
			return () => {
				holds--;
			};
		},
	};
	workspaceLifecycleThreadService.installMain?.(peer);
	link.handle("ping", nonce => nonce);
	const connected = Promise.withResolvers<WorkspaceProbe>();
	const test: WorkspaceTestGlobals = { ready: probe => connected.resolve(probe) };
	await link.call("boot", {
		[workspaceLifecycleThreadService.name]: workspaceLifecycleThreadService.globals?.(peer),
		test,
	});
	return {
		probe: await connected.promise,
		holds: () => holds,
		/** The worker died: close its link, terminate it, and run main's worker-exit cleanup. */
		exit: async (reason: string) => {
			link.close(reason);
			await worker.terminate();
			workspaceLifecycleThreadService.onWorkerExit?.(peer, reason);
		},
	};
}

function leaseInit(dir: string, agentId: string): Omit<WorkspaceLease, "generation" | "ownerPid" | "state"> {
	return {
		agentId,
		keepAlive: true,
		repoRoot: dir,
		workspaceDir: dir,
		baseDir: dir,
		backend: "fixture",
		baseline: { commit: "base", source: "working-tree", nested: "source-working-copy" },
	};
}

/** Register a main-owned lease whose release runs the gates, then release it explicitly. */
function gatedRelease(init: Omit<WorkspaceLease, "generation" | "ownerPid" | "state">) {
	const service = getWorkspaceLifecycleService();
	const generation = service.register(init, async () => {
		const hold = await service.runGates(init.agentId, generation);
		service.update(init.agentId, generation, hold ? { state: "held", holdReason: hold } : { state: "released" });
	});
	service.runFinished(init.agentId, generation);
	return service.release(init.agentId, { generation, reason: "test" });
}

function recordingGate(
	runs: string[],
	label: string,
	verdict: { ok: true } | { ok: false; hold: string } = { ok: true },
): WorkspaceReleaseGate {
	return async () => {
		runs.push(label);
		return verdict;
	};
}

it("main gates and inventories worker-owned leases, refuses stale generations, and holds a dead worker's workspace", async () => {
	using temp = TempDir.createSync("@omp-workspace-thread-");
	const fixture = await startWorker("w1");
	const service = getWorkspaceLifecycleService();
	const id = `WorkerLease-${path.basename(temp.path())}`;
	const seen: string[] = [];
	const stop = service.registerReleaseGate(async lease => {
		seen.push(lease.agentId);
		return { ok: false, hold: "main plugin preservation hold" };
	});
	try {
		const { probe } = fixture;
		const init = leaseInit(temp.path(), id);
		const first = await probe.register(init);
		expect(first.threadId).toBeGreaterThan(0);
		expect(fixture.holds()).toBe(1);
		expect((await probe.list()).find(lease => lease.agentId === id)).toEqual(
			service.list().find(lease => lease.agentId === id),
		);
		expect(await service.release(id, { generation: first.generation, reason: "busy" })).toMatchObject({
			status: "refused",
			reason: "active-owner: capture in progress",
		});
		await probe.finish(id, first.generation);
		expect(await service.release(id, { generation: "stale", reason: "stale" })).toMatchObject({
			status: "refused",
			reason: "stale-generation",
		});
		expect(await service.release(id, { generation: first.generation, reason: "preserve" })).toMatchObject({
			status: "held",
			reason: "main plugin preservation hold",
		});
		expect(seen).toEqual([id]);
		expect(fixture.holds()).toBe(0);
		expect(await Bun.file(path.join(temp.path(), WORKSPACE_LEASE_FILE)).json()).toMatchObject({
			generation: first.generation,
			state: "held",
		});
		const second = await probe.register(init);
		expect(await service.release(id, { generation: first.generation, reason: "old" })).toMatchObject({
			status: "refused",
			reason: "stale-generation",
		});
		const persisted = Promise.withResolvers<void>();
		const holdAtShutdown = service.holdAtShutdown.bind(service);
		// Observe the actual atomic sidecar completion, not a timer or filesystem poll.
		const hold = spyOn(service, "holdAtShutdown").mockImplementation(async (...args) => {
			await holdAtShutdown(...args);
			persisted.resolve();
		});
		try {
			await fixture.exit("test crash");
			await persisted.promise;
			expect(await Bun.file(path.join(temp.path(), WORKSPACE_LEASE_FILE)).json()).toMatchObject({
				generation: second.generation,
				state: "held",
				holdReason: "workspace owner worker exited: test crash",
			});
		} finally {
			hold.mockRestore();
		}
	} finally {
		stop();
		await fixture.exit("done");
		await fs.rm(path.join(temp.path(), WORKSPACE_LEASE_FILE), { force: true });
	}
}, 30_000);

it("runs a gate the same extension registered on main and in a worker once per release", async () => {
	using temp = TempDir.createSync("@omp-workspace-thread-");
	const fixture = await startWorker("w-same-extension");
	const extension = path.join(temp.path(), "workspace-lifecycle.ts");
	const runs: string[] = [];
	const stopMain = getWorkspaceLifecycleHost(extension).registerReleaseGate(recordingGate(runs, "main"));
	try {
		await fixture.probe.registerGate(extension, recordingGate(runs, "worker"));
		expect(await gatedRelease(leaseInit(temp.path(), `SameExtension-${path.basename(temp.path())}`))).toMatchObject({
			status: "released",
		});
		expect(runs).toEqual(["main"]);
	} finally {
		stopMain();
		await fixture.exit("done");
	}
}, 30_000);

it("runs and honors the hold of a gate from an extension only a worker loaded", async () => {
	using temp = TempDir.createSync("@omp-workspace-thread-");
	const fixture = await startWorker("w-worker-only");
	const runs: string[] = [];
	try {
		await fixture.probe.registerGate(
			path.join(temp.path(), "project-local-extension.ts"),
			recordingGate(runs, "worker-only", { ok: false, hold: "worker-only extension hold" }),
		);
		expect(await gatedRelease(leaseInit(temp.path(), `WorkerOnly-${path.basename(temp.path())}`))).toMatchObject({
			status: "held",
			reason: "worker-only extension hold",
		});
		expect(runs).toEqual(["worker-only"]);
		expect(await Bun.file(path.join(temp.path(), WORKSPACE_LEASE_FILE)).json()).toMatchObject({
			state: "held",
			holdReason: "worker-only extension hold",
		});
	} finally {
		await fixture.exit("done");
	}
}, 30_000);

it("hands an extension's gate to the worker once every main registration of it unsubscribed", async () => {
	using temp = TempDir.createSync("@omp-workspace-thread-");
	const fixture = await startWorker("w-takeover");
	const extension = path.join(temp.path(), "workspace-lifecycle.ts");
	const init = leaseInit(temp.path(), `Takeover-${path.basename(temp.path())}`);
	const runs: string[] = [];
	const mainHost = getWorkspaceLifecycleHost(extension);
	const stopMainA = mainHost.registerReleaseGate(recordingGate(runs, "main-a"));
	const stopMainB = mainHost.registerReleaseGate(recordingGate(runs, "main-b"));
	try {
		await fixture.probe.registerGate(extension, recordingGate(runs, "worker"));
		await gatedRelease(init);
		// Same-thread registrations of one extension all run, as without workers.
		expect(runs.splice(0)).toEqual(["main-a", "main-b"]);
		stopMainA();
		await gatedRelease(init);
		expect(runs.splice(0)).toEqual(["main-b"]);
		stopMainB();
		expect(await gatedRelease(init)).toMatchObject({ status: "released" });
		expect(runs.splice(0)).toEqual(["worker"]);
	} finally {
		stopMainA();
		stopMainB();
		await fixture.exit("done");
	}
}, 30_000);

it("hands an extension's gate back to main when the worker running it exits", async () => {
	using temp = TempDir.createSync("@omp-workspace-thread-");
	const fixture = await startWorker("w-exit");
	const extension = path.join(temp.path(), "workspace-lifecycle.ts");
	const init = leaseInit(temp.path(), `WorkerExit-${path.basename(temp.path())}`);
	const runs: string[] = [];
	await fixture.probe.registerGate(extension, recordingGate(runs, "worker"));
	const stopMain = getWorkspaceLifecycleHost(extension).registerReleaseGate(recordingGate(runs, "main"));
	try {
		await gatedRelease(init);
		expect(runs.splice(0)).toEqual(["worker"]);
		await fixture.exit("worker exited");
		expect(await gatedRelease(init)).toMatchObject({ status: "released" });
		expect(runs.splice(0)).toEqual(["main"]);
	} finally {
		stopMain();
		await fixture.exit("done");
	}
}, 30_000);
