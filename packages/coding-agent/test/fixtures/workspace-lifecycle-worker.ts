import { type MessagePort, parentPort, threadId } from "node:worker_threads";
import {
	getWorkspaceLifecycleHost,
	getWorkspaceLifecycleService,
	type WorkspaceLease,
	type WorkspaceReleaseGate,
} from "../../src/task/workspace-lifecycle";
import type { WorkerThreadLink } from "../../src/thread/protocol";
import { type RpcFrame, RpcLink } from "../../src/thread/rpc";
import { workspaceLifecycleThreadService } from "../../src/thread/services/workspace-lifecycle";

export interface WorkspaceProbe {
	register(
		init: Omit<WorkspaceLease, "generation" | "ownerPid" | "state">,
	): Promise<{ generation: string; threadId: number }>;
	finish(id: string, generation: string): Promise<void>;
	list(): Promise<WorkspaceLease[]>;
	/**
	 * Register a worker-local gate through extension `extension`'s plugin-facing host, as that
	 * extension's session_start would in this thread; it delegates to `gate`. Resolves its unsubscribe.
	 */
	registerGate(extension: string, gate: WorkspaceReleaseGate): Promise<() => Promise<void>>;
}
export interface WorkspaceTestGlobals {
	ready(probe: WorkspaceProbe): void;
}
const port = parentPort;
if (!port) throw new Error("fixture must run in a worker");
port.once("message", (init: { sync: MessagePort; counter: SharedArrayBuffer }) => {
	const link: WorkerThreadLink = new RpcLink(
		{
			send: frame => port.postMessage(frame),
			listen: handler => {
				const listener = (frame: RpcFrame): void => handler(frame);
				port.on("message", listener);
				return () => port.off("message", listener);
			},
		},
		{ name: "workspace-fixture", sync: { port: init.sync, counter: new Int32Array(init.counter) } },
	);
	workspaceLifecycleThreadService.installWorker?.(link);
	link.handle("ping", nonce => nonce);
	link.handle("boot", async globals => {
		await workspaceLifecycleThreadService.boot?.(globals[workspaceLifecycleThreadService.name], link);
		const test = globals.test as WorkspaceTestGlobals;
		const service = getWorkspaceLifecycleService();
		await test.ready({
			register: async init => {
				const generation = service.register(init, async () => {
					const hold = await service.runGates(init.agentId, generation);
					service.update(
						init.agentId,
						generation,
						hold ? { state: "held", holdReason: hold } : { state: "released" },
					);
				});
				return { generation, threadId };
			},
			finish: async (id, generation) => service.runFinished(id, generation),
			list: async () => getWorkspaceLifecycleHost().list(),
			registerGate: async (extension, gate) => {
				const stop = getWorkspaceLifecycleHost(extension).registerReleaseGate((lease, ctx) => gate(lease, ctx));
				return async () => stop();
			},
		});
	});
	port.postMessage({ ready: true });
});
