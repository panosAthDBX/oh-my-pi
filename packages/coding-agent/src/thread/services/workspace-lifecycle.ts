import {
	getWorkspaceLifecycleService,
	installWorkspaceLifecycleService,
	WorkspaceLifecycleService,
	type WorkspaceLifecycleServiceApi,
	type WorkspaceLease,
	type WorkspaceReleaseGate,
	type WorkspaceReleaseResult,
} from "../../task/workspace-lifecycle";
import { trackLateCleanup } from "../../utils/late-cleanup";
import type { RemoteObject, RpcMethod, RpcServe } from "../rpc";
import type { ThreadService, WorkerPeer } from "../services";

const KIND = "workspace-lifecycle";
type LeaseInit = Omit<WorkspaceLease, "generation" | "ownerPid" | "state">;
type ReleaseOptions = { generation: string; reason: string };
type WorkspaceProtocol = {
	hostVersion: RpcMethod<[], string>;
	list: RpcMethod<[], WorkspaceLease[]>;
	hasLiveWorkspace: RpcMethod<[agentId: string], boolean>;
	hasGatedRelease: RpcMethod<[agentId: string], boolean>;
	inFlightGatedReleases: RpcMethod<[], Array<{ agentId: string; settled: () => Promise<void> }>>;
	whenSettled: RpcMethod<[agentId: string, generation: string], void>;
	holdAtShutdown: RpcMethod<[agentId: string, reason: string], void>;
	release: RpcMethod<[agentId: string, opts: ReleaseOptions], WorkspaceReleaseResult>;
	register: RpcMethod<[init: LeaseInit, release: () => Promise<void>, ownerStartToken?: string], string>;
	update: RpcMethod<[agentId: string, generation: string, patch: Partial<WorkspaceLease>], void>;
	runFinished: RpcMethod<[agentId: string, generation: string], void>;
	runGates: RpcMethod<[agentId: string, generation: string], string | undefined>;
};

/** All lease mutations and plugin gates run on main, including nested worker spawns. */
class RemoteWorkspaceLifecycle implements WorkspaceLifecycleServiceApi {
	readonly apiVersion = 1 as const;
	constructor(readonly remote: RemoteObject<WorkspaceProtocol>) {}
	get hostVersion(): string {
		return this.remote.callSync("hostVersion");
	}
	list(): WorkspaceLease[] {
		return this.remote.callSync("list");
	}
	hasLiveWorkspace(agentId: string): boolean {
		return this.remote.callSync("hasLiveWorkspace", agentId);
	}
	hasGatedRelease(agentId: string): boolean {
		return this.remote.callSync("hasGatedRelease", agentId);
	}
	inFlightGatedReleases(): Array<{ agentId: string; settled: Promise<void> }> {
		return this.remote.callSync("inFlightGatedReleases").map(entry => ({
			agentId: entry.agentId,
			settled: entry.settled(),
		}));
	}
	whenSettled(agentId: string, generation: string): Promise<void> {
		return this.remote.call("whenSettled", agentId, generation);
	}
	holdAtShutdown(agentId: string, reason: string): Promise<void> {
		return this.remote.call("holdAtShutdown", agentId, reason);
	}
	/**
	 * Every thread loads the same extensions, so main's instance already registered this gate.
	 * Forwarding a worker's copy would run each gate (and its preservation) once per live worker.
	 */
	registerReleaseGate(_gate: WorkspaceReleaseGate): () => void {
		return () => {};
	}
	release(agentId: string, opts: ReleaseOptions): Promise<WorkspaceReleaseResult> {
		return this.remote.call("release", agentId, opts);
	}
	register(init: LeaseInit, release: () => Promise<void>, ownerStartToken?: string): string {
		return this.remote.callSync("register", init, release, ownerStartToken);
	}
	update(agentId: string, generation: string, patch: Partial<WorkspaceLease>): void {
		this.remote.callSync("update", agentId, generation, patch);
	}
	runFinished(agentId: string, generation: string): void {
		this.remote.callSync("runFinished", agentId, generation);
	}
	runGates(agentId: string, generation: string): Promise<string | undefined> {
		return this.remote.call("runGates", agentId, generation);
	}
}

interface OwnedLease {
	service: WorkspaceLifecycleService;
	agentId: string;
	generation: string;
	release: () => void;
}
interface WorkerOwnership {
	leases: Map<string, OwnedLease>;
}
const owners = new Map<WorkerPeer["threadId"], WorkerOwnership>();

export const workspaceLifecycleThreadService: ThreadService = {
	name: KIND,
	installMain(peer): void {
		const owner: WorkerOwnership = { leases: new Map() };
		owners.set(peer.threadId, owner);
		const serve: RpcServe<WorkspaceLifecycleService, WorkspaceProtocol> = {
			hostVersion: service => service.hostVersion,
			list: service => service.list(),
			hasLiveWorkspace: (service, id) => service.hasLiveWorkspace(id),
			hasGatedRelease: (service, id) => service.hasGatedRelease(id),
			inFlightGatedReleases: service =>
				service.inFlightGatedReleases().map(entry => ({
					agentId: entry.agentId,
					settled: () => entry.settled,
				})),
			whenSettled: (service, id, generation) => service.whenSettled(id, generation),
			holdAtShutdown: (service, id, reason) => service.holdAtShutdown(id, reason),
			release: (service, id, opts) => service.release(id, opts),
			register: (service, init, release, token) => {
				const generation = service.register(init, release, token);
				owner.leases.get(init.agentId)?.release();
				owner.leases.set(init.agentId, {
					service,
					agentId: init.agentId,
					generation,
					release: peer.retain?.() ?? (() => {}),
				});
				void service.whenSettled(init.agentId, generation).then(() => {
					const lease = owner.leases.get(init.agentId);
					if (lease?.generation !== generation) return;
					owner.leases.delete(init.agentId);
					lease.release();
				});
				return generation;
			},
			update: (service, id, generation, patch) => service.update(id, generation, patch),
			runFinished: (service, id, generation) => service.runFinished(id, generation),
			runGates: (service, id, generation) => service.runGates(id, generation),
		};
		peer.link.exportClass(
			KIND,
			(value): value is WorkspaceLifecycleService => value instanceof WorkspaceLifecycleService,
			serve,
		);
	},
	globals(): WorkspaceLifecycleServiceApi {
		return getWorkspaceLifecycleService();
	},
	onWorkerExit(peer, reason): void {
		const owner = owners.get(peer.threadId);
		owners.delete(peer.threadId);
		if (!owner) return;
		for (const lease of owner.leases.values()) {
			// A delayed worker exit must not hold a successor that reused its id.
			const current = lease.service.list().find(item => item.agentId === lease.agentId);
			if (current?.generation === lease.generation) {
				trackLateCleanup(
					lease.service
						.holdAtShutdown(lease.agentId, `workspace owner worker exited: ${reason}`)
						.finally(lease.release),
					{ agentId: lease.agentId, resource: "isolation" },
				);
			} else lease.release();
		}
		owner.leases.clear();
	},
	installWorker(link): void {
		link.importClass<RemoteWorkspaceLifecycle, WorkspaceProtocol>(
			KIND,
			remote => new RemoteWorkspaceLifecycle(remote),
		);
	},
	boot(globals): void {
		if (!(globals instanceof RemoteWorkspaceLifecycle))
			throw new Error("Malformed workspace lifecycle thread globals");
		installWorkspaceLifecycleService(globals);
	},
};
