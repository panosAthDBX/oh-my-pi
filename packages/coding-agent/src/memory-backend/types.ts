/**
 * Memory backend abstraction.
 *
 * Backends are mutually exclusive — `await resolveMemoryBackend(settings)` returns
 * exactly one. Implementations MUST be self-contained: they own the per-session
 * state they create in `start()` and tear it down on `clear()`.
 */

import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { ModelRegistry } from "../config/model-registry";
import type { Settings } from "../config/settings";
import type { HindsightSessionState } from "../hindsight/state";
import type { MnemopiSessionState } from "../mnemopi/state";
import type { AgentSession } from "../session/agent-session";

export type MemoryBackendId = "off" | "local" | "hindsight" | "mnemopi" | "supermemory";

export interface MemoryBackendStatus {
	backend: MemoryBackendId;
	active: boolean;
	writable: boolean;
	searchable: boolean;
	scope?: string;
	retainBank?: string;
	recallBanks?: string[];
	workingCount?: number;
	episodicCount?: number;
	tripleCount?: number;
	lastMemory?: string;
	lastRecall?: boolean;
	database?: string;
	message?: string;
	error?: string;
}

export interface MemoryBackendSearchOptions {
	limit?: number;
	/** Best-effort abort signal. Backends may only observe it before/after an underlying recall call. */
	signal?: AbortSignal;
}

export interface MemoryBackendSearchItem {
	id?: string;
	content: string;
	source?: string;
	timestamp?: string;
	score?: number;
}

export interface MemoryBackendSearchResult {
	backend: MemoryBackendId;
	query: string;
	count: number;
	items: MemoryBackendSearchItem[];
	message?: string;
}

export interface MemoryBackendSaveInput {
	content: string;
	context?: string;
	source?: string;
	importance?: number;
}

export interface MemoryBackendSaveResult {
	backend: MemoryBackendId;
	stored: number;
	ids?: string[];
	queued?: boolean;
	message?: string;
}

export interface MemoryBackendOperationContext {
	agentDir: string;
	cwd: string;
	session?: AgentSession;
}

export interface MemoryRuntimeContext {
	status(): Promise<MemoryBackendStatus>;
	search(query: string, options?: MemoryBackendSearchOptions): Promise<MemoryBackendSearchResult>;
	save(input: string | MemoryBackendSaveInput): Promise<MemoryBackendSaveResult>;
}

export interface MemoryBackendStartOptions {
	session: AgentSession;
	settings: Settings;
	modelRegistry: ModelRegistry;
	agentDir: string;
	taskDepth: number;
	parentHindsightSessionState?: HindsightSessionState;
	parentMnemopiSessionState?: MnemopiSessionState;
}

export interface MemoryBackendBeforeAgentStartOptions {
	/** Best-effort cancellation for this turn's recall work. */
	signal?: AbortSignal;
	/** Monotonically increasing AgentSession turn generation. */
	generation?: number;
	/** True only while the originating turn is still eligible to mutate its prompt. */
	isCurrent?: () => boolean;
}

/**
 * Context identifying the prompt that completed preflight and is about to enter
 * the agent loop. This is intentionally identical to the context supplied to
 * {@link MemoryBackend.beforeAgentStartPrompt}.
 */
export type MemoryBackendCommitAgentStartOptions = MemoryBackendBeforeAgentStartOptions;

/**
 * A commit prepared asynchronously during prompt preflight. Its `commit` method
 * runs synchronously only after agent-core has accepted the prompt, so rejected
 * busy/validation calls cannot consume turn-local memory state.
 */
export interface MemoryBackendPreparedAgentStartCommit {
	commit(): void;
}

export interface MemoryBackend {
	readonly id: MemoryBackendId;

	/**
	 * Wire any background work or session subscriptions for this backend.
	 *
	 * Called once per agent session at startup. Implementations MUST be
	 * non-throwing: failures should be logged and swallowed so a misconfigured
	 * memory backend cannot break the agent loop.
	 */
	start(options: MemoryBackendStartOptions): void | Promise<void>;

	/**
	 * Reset per-transcript tracking after this session switches to a new
	 * transcript. Return true when developer instructions need rebuilding.
	 */
	resetSession?(session: AgentSession): boolean | void | Promise<boolean | undefined> | Promise<void>;

	/** Flush transcript-scoped work before the session manager replaces or deletes the current transcript. */
	beforeTranscriptReplace?(session: AgentSession): void | Promise<void>;

	/** Release session-scoped resources during AgentSession disposal. */
	disposeSession?(session: AgentSession): void | Promise<void>;

	/**
	 * Markdown injected as the system-prompt append section.
	 * Returned on every prompt rebuild via `refreshBaseSystemPrompt()`.
	 */
	buildDeveloperInstructions(
		agentDir: string,
		settings: Settings,
		session?: AgentSession,
	): Promise<string | undefined>;

	/** Wipe all persisted state for this backend (slash `/memory clear`). */
	clear(agentDir: string, cwd: string, session?: AgentSession): Promise<void>;

	/** Force consolidation/retain to happen now (slash `/memory enqueue`). */
	enqueue(agentDir: string, cwd: string, session?: AgentSession): Promise<void>;

	/** Structured state for UI, slash commands, and extensions. */
	status?(context: MemoryBackendOperationContext): Promise<MemoryBackendStatus>;

	/** Explicit user-facing semantic/lexical search. */
	search?(
		context: MemoryBackendOperationContext,
		query: string,
		options?: MemoryBackendSearchOptions,
	): Promise<MemoryBackendSearchResult>;

	/** Explicit user-facing save operation. */
	save?(context: MemoryBackendOperationContext, input: MemoryBackendSaveInput): Promise<MemoryBackendSaveResult>;

	/** Render backend-specific memory statistics as markdown (`/memory stats`). */
	stats?(agentDir: string, cwd: string, session?: AgentSession): Promise<string | undefined>;

	/** Render backend-specific memory diagnostics as markdown (`/memory diagnose`). */
	diagnose?(agentDir: string, cwd: string, session?: AgentSession): Promise<string | undefined>;
	/**
	 * Optional hook to inject a backend-specific block into the current turn's
	 * system prompt before the agent starts generating.
	 *
	 * This is the only place a backend can affect the very first answer of a
	 * fresh session. The returned text is appended to the already-built base
	 * system prompt for this turn only; callers may separately cache it and
	 * surface it through `buildDeveloperInstructions()` on later rebuilds.
	 */
	beforeAgentStartPrompt?(
		session: AgentSession,
		promptText: string,
		options?: MemoryBackendBeforeAgentStartOptions,
	): Promise<string | undefined>;

	/**
	 * Prepare a turn-local commit after prompt preflight. Returning `false` means
	 * the staged request is no longer current and the caller must rebuild recall
	 * before retrying the same prompt. A returned token is committed synchronously
	 * from agent-core's accepted-admission callback. `void` preserves compatibility
	 * for backends with no turn-local state to commit.
	 */
	commitBeforeAgentStartPrompt?(
		session: AgentSession,
		promptText: string,
		options?: MemoryBackendCommitAgentStartOptions,
	): Promise<MemoryBackendPreparedAgentStartCommit | false | undefined> | Promise<void>;

	/**
	 * Optional hook to splice extra context into a compaction summarization.
	 *
	 * Called from the compaction call site before the LLM summary is requested.
	 * Returning a string appends one entry to the compaction's `extraContext`
	 * list (which becomes part of the summarization prompt). Return `undefined`
	 * to inject nothing — the local backend takes this branch because its
	 * summary is already part of the system prompt.
	 */
	preCompactionContext?(
		messages: AgentMessage[],
		settings: Settings,
		session?: AgentSession,
	): Promise<string | undefined>;
}
