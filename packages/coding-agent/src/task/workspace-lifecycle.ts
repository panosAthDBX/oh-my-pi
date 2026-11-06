/**
 * Host companion for workspace-lifecycle plugins (`ctx.workspaceLifecycle`, apiVersion 1).
 *
 * One process-global lease table for isolated task workspaces. The isolation
 * runner registers a lease once a workspace is materialised and routes every
 * teardown through {@link WorkspaceLifecycleService.runGates}: a gate hold,
 * gate error, capture failure or native stop failure retains the workspace
 * and settles the lease as `held` instead of deleting it. Each lease is also
 * persisted next to the workspace (`<baseDir>/.omp-workspace-lease.json`) so
 * a restarted process can identify it.
 */
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { logger, Snowflake, VERSION } from "@oh-my-pi/pi-utils";
import { AgentLifecycleManager } from "../registry/agent-lifecycle";
import { AgentRegistry } from "../registry/agent-registry";

export type WorkspaceLeaseState = "active" | "idle" | "parked" | "releasing" | "released" | "held";

export interface WorkspaceLease {
	agentId: string;
	generation: string;
	ownerPid: number;
	keepAlive: boolean;
	state: WorkspaceLeaseState;
	repoRoot: string;
	workspaceDir: string;
	baseDir: string;
	backend: string;
	baseline: {
		commit: string;
		requested?: string;
		source: "working-tree" | "explicit-commit";
		nested: "source-working-copy";
	};
	capture?: { patchPath?: string; branchName?: string; nestedPatchPaths?: string[] };
	holdReason?: string;
	retainedDir?: string;
}

/** `lifecycle` = native release without an explicit caller (idle release, one-shot completion, process shutdown). */
export type WorkspaceReleaseTrigger = "explicit" | "lifecycle";

export type WorkspaceReleaseGate = (
	lease: WorkspaceLease,
	ctx: { trigger: WorkspaceReleaseTrigger; reason: string },
) => Promise<{ ok: true; receipt?: string } | { ok: false; hold: string }>;

export interface WorkspaceReleaseResult {
	status: "released" | "already-released" | "held" | "refused";
	reason?: string;
	lease?: WorkspaceLease;
}

export interface WorkspaceLifecycleHost {
	readonly apiVersion: 1;
	readonly hostVersion: string;
	list(): WorkspaceLease[];
	release(agentId: string, opts: { generation: string; reason: string }): Promise<WorkspaceReleaseResult>;
	registerReleaseGate(gate: WorkspaceReleaseGate): () => void;
}

/** Lease sidecar written into the isolation base dir; travels with retain renames. */
export const WORKSPACE_LEASE_FILE = ".omp-workspace-lease.json";

interface LeaseEntry {
	lease: WorkspaceLease;
	/** The runner's memoized capture+gate+teardown. */
	release: () => Promise<void>;
	trigger: WorkspaceReleaseTrigger;
	reason: string;
	explicit?: Promise<WorkspaceReleaseResult>;
	persisting: Promise<void>;
	unsubscribe?: () => void;
	ownerStartToken?: string;
	/** The runner's post-run capture (branch commit, patch, nested patches) is still using the workspace. */
	runnerBusy: boolean;
	/** Every gate passed and teardown began: a shutdown hold would describe a workspace being removed. */
	tearingDown?: boolean;
	/** Resolves once the lease settles (`released` or `held`). */
	settled: PromiseWithResolvers<void>;
}

const STATUS_STATES = { running: "active", idle: "idle", parked: "parked" } as const;

function isSettled(lease: WorkspaceLease): boolean {
	return lease.state === "released" || lease.state === "held";
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

/** Main-owned service surface; workers install an RPC proxy with the same contract. */
export type WorkspaceLifecycleServiceApi = Pick<WorkspaceLifecycleService, keyof WorkspaceLifecycleService>;

export class WorkspaceLifecycleService implements WorkspaceLifecycleHost {
	readonly apiVersion = 1 as const;
	readonly hostVersion = VERSION;
	readonly #leases = new Map<string, LeaseEntry>();
	/**
	 * Registration order. `key` is the registering extension's resolved path and `origin` its
	 * thread (undefined = main): every thread loads the same extension, so one key's gates
	 * run only from the earliest origin still registered; unkeyed gates always run.
	 */
	readonly #gates: Array<{ gate: WorkspaceReleaseGate; key?: string; origin?: unknown }> = [];
	readonly #persisting = new Map<string, Promise<void>>();

	list(): WorkspaceLease[] {
		return [...this.#leases.values()].map(entry => structuredClone(entry.lease));
	}

	/** The lease is live (not releasing, released or held), so a revived lane still has its worktree. */
	hasLiveWorkspace(agentId: string): boolean {
		const lease = this.#leases.get(agentId)?.lease;
		return lease !== undefined && lease.state !== "releasing" && !isSettled(lease);
	}

	/** A gated release may still run for this lease: gates are registered and the lease is unsettled. */
	hasGatedRelease(agentId: string): boolean {
		const lease = this.#leases.get(agentId)?.lease;
		return this.#gates.length > 0 && lease !== undefined && !isSettled(lease);
	}

	/**
	 * Gated releases already in flight (`releasing`) and not owned by the agent
	 * lifecycle: explicit releases, whose agent was already released, and
	 * one-shot runs tearing down in their own run. `settled` resolves when the
	 * lease settles. A lease still before its release (active one-shot run) is
	 * not listed: shutdown cannot tell how long that run will take.
	 */
	inFlightGatedReleases(): Array<{ agentId: string; settled: Promise<void> }> {
		if (this.#gates.length === 0) return [];
		return [...this.#leases.values()]
			.filter(entry => entry.lease.state === "releasing")
			.map(entry => ({ agentId: entry.lease.agentId, settled: entry.settled.promise }));
	}

	/** Keep a worker owner alive until its exact lease's terminal sidecar is durable. */
	whenSettled(agentId: string, generation: string): Promise<void> {
		return this.#current(agentId, generation)?.settled.promise ?? Promise.resolve();
	}

	/**
	 * Process shutdown gave up on this lease's release: settle it as `held` and
	 * wait for the sidecar write. A late release then holds instead of removing,
	 * whether it is still before its gates or inside one (see {@link runGates}).
	 *
	 * Accepted gaps, left non-terminal for the engine's dead-owner (orphan)
	 * classification: a lease already tearing down keeps `releasing` (a partly
	 * removed workspace must not be described as held); a failed sidecar write is
	 * only logged; an exit between a late retain's rename and its lease update
	 * leaves the `.retained-*` sidecar with the previous paths. The tmp+rename
	 * write is atomic for a process exit, not for power loss (no fsync).
	 */
	async holdAtShutdown(agentId: string, holdReason: string): Promise<void> {
		const entry = this.#leases.get(agentId);
		if (!entry || entry.tearingDown || isSettled(entry.lease)) return;
		this.#setState(entry, { state: "held", holdReason });
		await entry.persisting;
	}

	registerReleaseGate(gate: WorkspaceReleaseGate, key?: string, origin?: unknown): () => void {
		// Re-registering the same gate from the same extension and thread is one registration, as before keys.
		const entry = this.#gates.find(item => item.gate === gate && item.key === key && item.origin === origin) ?? {
			gate,
			key,
			origin,
		};
		if (!this.#gates.includes(entry)) this.#gates.push(entry);
		return () => {
			const index = this.#gates.indexOf(entry);
			if (index >= 0) this.#gates.splice(index, 1);
		};
	}

	#activeGates(): WorkspaceReleaseGate[] {
		const owners = new Map<string, unknown>();
		return this.#gates
			.filter(({ key, origin }) => {
				if (key === undefined) return true;
				if (!owners.has(key)) owners.set(key, origin);
				return owners.get(key) === origin;
			})
			.map(entry => entry.gate);
	}

	release(agentId: string, opts: { generation: string; reason: string }): Promise<WorkspaceReleaseResult> {
		const entry = this.#leases.get(agentId);
		if (!entry) return Promise.resolve({ status: "refused", reason: "unknown-agent" });
		if (entry.lease.generation !== opts.generation) {
			return Promise.resolve({ status: "refused", reason: "stale-generation", lease: structuredClone(entry.lease) });
		}
		if (entry.explicit) {
			// Concurrent callers share the pending outcome; later callers see the settled state.
			return isSettled(entry.lease) ? Promise.resolve(this.#result(entry, true)) : entry.explicit;
		}
		if (isSettled(entry.lease)) return Promise.resolve(this.#result(entry, true));
		const lifecycle = AgentLifecycleManager.global();
		const ref = AgentRegistry.global().get(agentId);
		if (
			ref?.status === "running" ||
			lifecycle.isReviving(agentId) ||
			lifecycle.isParking(agentId) ||
			// A one-shot lease settles in its own run; unsettled means the run is still going.
			(!entry.lease.keepAlive && !lifecycle.has(agentId))
		) {
			return Promise.resolve({ status: "refused", reason: "active-owner", lease: structuredClone(entry.lease) });
		}
		if (entry.runnerBusy) {
			return Promise.resolve({
				status: "refused",
				reason: "active-owner: capture in progress",
				lease: structuredClone(entry.lease),
			});
		}
		entry.trigger = "explicit";
		entry.reason = opts.reason;
		this.#setState(entry, { state: "releasing" });
		entry.explicit = (async () => {
			try {
				// Tombstoned: the ref stays as a terminal row so it can never be revived.
				if (lifecycle.has(agentId)) await lifecycle.release(agentId, undefined, { tombstone: true });
				await entry.release();
			} catch (error) {
				if (!isSettled(entry.lease)) this.#setState(entry, { state: "held", holdReason: errorMessage(error) });
			}
			if (!isSettled(entry.lease)) this.#setState(entry, { state: "held", holdReason: "release did not settle" });
			await entry.persisting;
			return this.#result(entry, false);
		})();
		return entry.explicit;
	}

	#result(entry: LeaseEntry, repeated: boolean): WorkspaceReleaseResult {
		const lease = structuredClone(entry.lease);
		if (lease.state === "held") return { status: "held", reason: lease.holdReason, lease };
		return { status: repeated ? "already-released" : "released", lease };
	}

	// ── Runner-facing (not part of the plugin API) ─────────────────────────

	/** Record a freshly materialised workspace; returns the lease generation. */
	register(
		init: Omit<WorkspaceLease, "generation" | "ownerPid" | "state">,
		release: () => Promise<void>,
		ownerStartToken?: string,
	): string {
		const generation = Snowflake.next();
		const entry: LeaseEntry = {
			lease: { ...init, generation, ownerPid: process.pid, state: "active" },
			release,
			trigger: "lifecycle",
			reason: "lifecycle",
			persisting: Promise.resolve(),
			ownerStartToken,
			runnerBusy: true,
			settled: Promise.withResolvers<void>(),
		};
		const previous = this.#leases.get(init.agentId);
		previous?.unsubscribe?.();
		if (previous) void previous.persisting.then(() => previous.settled.resolve());
		this.#leases.set(init.agentId, entry);
		const registry = AgentRegistry.global();
		entry.unsubscribe = registry.onChange(event => {
			if (event.type !== "status_changed" || event.ref.id !== init.agentId) return;
			if (entry.lease.state === "releasing" || isSettled(entry.lease)) return;
			const state = STATUS_STATES[event.ref.status as keyof typeof STATUS_STATES];
			if (state && state !== entry.lease.state) this.#setState(entry, { state });
		});
		this.#persist(entry);
		return generation;
	}

	/** Merge runner facts (capture, retained paths) into the lease. Ignores superseded generations. */
	update(agentId: string, generation: string, patch: Partial<WorkspaceLease>): void {
		const entry = this.#current(agentId, generation);
		if (entry) this.#setState(entry, patch);
	}

	/** The runner's post-run capture finished; explicit release may proceed. */
	runFinished(agentId: string, generation: string): void {
		const entry = this.#current(agentId, generation);
		if (entry) entry.runnerBusy = false;
	}

	/**
	 * Run every active registered gate (see {@link registerReleaseGate}) for this lease. Returns the hold
	 * reason, or undefined when all gates pass (or none are registered). A superseded
	 * generation holds: its workspace is no longer described by any lease.
	 */
	async runGates(agentId: string, generation: string): Promise<string | undefined> {
		const entry = this.#current(agentId, generation);
		if (!entry) return "lease superseded";
		// Settled before its gates (a shutdown hold during capture): retain, never re-open as `releasing`.
		if (isSettled(entry.lease)) return entry.lease.holdReason ?? "lease settled before release";
		this.#setState(entry, { state: "releasing" });
		for (const gate of this.#activeGates()) {
			try {
				const verdict = await gate(structuredClone(entry.lease), { trigger: entry.trigger, reason: entry.reason });
				if (this.#current(agentId, generation) !== entry) return "lease superseded";
				if (isSettled(entry.lease)) return entry.lease.holdReason ?? "lease settled during release";
				if (!verdict.ok) return verdict.hold || "release gate hold";
			} catch (error) {
				return `release gate failed: ${errorMessage(error)}`;
			}
		}
		// Drain every write queued while waiting; permission belongs only to this exact, unsettled lease.
		let persisting: Promise<void>;
		do {
			persisting = entry.persisting;
			await persisting;
			if (this.#current(agentId, generation) !== entry) return "lease superseded";
			if (isSettled(entry.lease)) return entry.lease.holdReason ?? "lease settled during release";
		} while (persisting !== entry.persisting);
		entry.tearingDown = true;
		return undefined;
	}

	#current(agentId: string, generation: string): LeaseEntry | undefined {
		const entry = this.#leases.get(agentId);
		return entry?.lease.generation === generation ? entry : undefined;
	}

	#setState(entry: LeaseEntry, patch: Partial<WorkspaceLease>): void {
		Object.assign(entry.lease, patch);
		this.#persist(entry);
		if (isSettled(entry.lease)) {
			entry.unsubscribe?.();
			entry.unsubscribe = undefined;
			// After the settled sidecar write, so a shutdown waiting on it never exits first.
			void entry.persisting.then(() => entry.settled.resolve());
		}
	}

	/** Atomic tmp+rename writes, ordered per lease and destination across slot reuse. */
	#persist(entry: LeaseEntry): void {
		// An old explicit release may finish after register replaced it; it cannot publish new state.
		if (this.#leases.get(entry.lease.agentId) !== entry) return;
		const snapshot = structuredClone(entry.lease);
		if (snapshot.state === "released") return;
		const ref = AgentRegistry.global().get(snapshot.agentId);
		const record = {
			...snapshot,
			sessionFile: ref?.sessionFile ?? undefined,
			parentAgentId: ref?.parentId,
			ownerStartToken: entry.ownerStartToken,
		};
		const file = path.resolve(snapshot.baseDir, WORKSPACE_LEASE_FILE);
		const persisting = Promise.all([entry.persisting, this.#persisting.get(file)]).then(async () => {
			const tmp = `${file}.${process.pid}.${Snowflake.next()}.tmp`;
			try {
				await Bun.write(tmp, JSON.stringify(record, null, 2), { createPath: false });
				await fs.rename(tmp, file);
			} catch (error) {
				await fs.rm(tmp, { force: true }).catch(() => undefined);
				logger.warn("workspace lease persist failed", { agentId: snapshot.agentId, error: errorMessage(error) });
			}
		});
		entry.persisting = persisting;
		this.#persisting.set(file, persisting);
		void persisting.then(() => {
			if (this.#persisting.get(file) === persisting) this.#persisting.delete(file);
		});
	}
}

let globalService: WorkspaceLifecycleServiceApi | undefined;
let globalHosts = new Map<string | undefined, WorkspaceLifecycleHost>();

/** Worker bootstrap installs main's service before creating any sessions. */
export function installWorkspaceLifecycleService(service: WorkspaceLifecycleServiceApi): void {
	globalService = service;
	globalHosts = new Map();
}

export function getWorkspaceLifecycleService(): WorkspaceLifecycleServiceApi {
	globalService ??= new WorkspaceLifecycleService();
	return globalService;
}

/**
 * Frozen plugin-facing view (`ctx.workspaceLifecycle`): the apiVersion 1 surface only, one identity per
 * extension (`extension` = its resolved path) per thread. Gates registered through an extension's view
 * are keyed by it, so other threads loading the same extension do not run it again; without an
 * extension, gates are unkeyed and always run.
 */
export function getWorkspaceLifecycleHost(extension?: string): WorkspaceLifecycleHost {
	let host = globalHosts.get(extension);
	if (host) return host;
	const service = getWorkspaceLifecycleService();
	host = Object.freeze({
		apiVersion: service.apiVersion,
		hostVersion: service.hostVersion,
		list: () => service.list(),
		release: (agentId: string, opts: { generation: string; reason: string }) => service.release(agentId, opts),
		registerReleaseGate: (gate: WorkspaceReleaseGate) => service.registerReleaseGate(gate, extension),
	});
	globalHosts.set(extension, host);
	return host;
}
