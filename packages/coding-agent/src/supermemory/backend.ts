import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import { logger, prompt } from "@oh-my-pi/pi-utils";
import type { Settings } from "../config/settings";
import { extractHindsightMessage } from "../hindsight/transcript";
import type {
	MemoryBackend,
	MemoryBackendBeforeAgentStartOptions,
	MemoryBackendCommitAgentStartOptions,
	MemoryBackendOperationContext,
	MemoryBackendPreparedAgentStartCommit,
	MemoryBackendSaveInput,
	MemoryBackendSaveResult,
	MemoryBackendSearchOptions,
	MemoryBackendSearchResult,
	MemoryBackendStartOptions,
	MemoryBackendStatus,
} from "../memory-backend/types";
import type { AgentSession } from "../session/agent-session";
import { SupermemoryClient, type SupermemorySearchItem } from "./client";
import {
	isSupermemoryConfigured,
	loadSupermemoryConfig,
	resolveSupermemoryContainerTag,
	type SupermemoryConfig,
} from "./config";
import { escapeSupermemoryXmlText } from "./content";
import instructions from "./instructions.md" with { type: "text" };
import profileContextTemplate from "./profile-context.md" with { type: "text" };
import recallContextTemplate from "./recall-context.md" with { type: "text" };

interface SupermemoryScopeSnapshot {
	client: SupermemoryClient;
	config: SupermemoryConfig & { apiKey: string };
	containerTag: string;
	coordinatorKey: string;
	generation: number;
}

interface ClearAdmissionWatermark {
	transcriptId: string;
	lifecycleGeneration: number;
	userTurnCount: number;
}

interface SupermemoryContainerState {
	states: Set<SupermemorySessionState>;
	documentWrites: Set<Promise<unknown>>;
	clearing?: Promise<void>;
	clearAdmissionWatermarks?: Map<SupermemorySessionState, ClearAdmissionWatermark>;
	/** Successful clears remain visible to states joining this in-process coordinator. */
	successfulClearGeneration: number;
}

interface SupermemorySessionState {
	session: AgentSession;
	client: SupermemoryClient;
	settings: Settings;
	config: SupermemoryConfig & { apiKey: string };
	containerTag: string;
	coordinatorKey: string;
	ready: Promise<void>;
	automatic: boolean;
	lifecycleGeneration: number;
	recallInvalidationGeneration: number;
	scopeTransition?: Promise<void>;
	pendingTranscriptResume: boolean;
	retention: {
		scopeGeneration: number;
		containerTag: string;
		lastRetainedTurn: number;
	};
	pendingRecall?: {
		snippet?: string;
		promptText: string;
		generation?: number;
		signal?: AbortSignal;
		scope: SupermemoryScopeSnapshot;
	};
	disposed: boolean;
	/** Latest successful clear boundary observed for each coordinator scope. */
	observedSuccessfulClearGenerations: Map<string, number>;
	hasRecalledForFirstTurn: boolean;
	lastRecallSnippet?: string;
	lastError?: string;
	lastMemoryId?: string;
	lastMemoryStatus?: string;
	lastSearchCount: number;
	lastSearchAt?: number;
	retainInFlight?: Promise<void>;
	retainPending: boolean;
	retainForceTail: boolean;
	/** Monotonic marker prevents an older completed flush from clearing a new force request. */
	retainForceTailEpoch: number;
	unsubscribe?: () => void;
}

const sessionStates = new WeakMap<AgentSession, SupermemorySessionState>();
const containerStates = new Map<string, SupermemoryContainerState>();
const RETENTION_SHUTDOWN_TIMEOUT_MS = 2_000;
const MAX_AUTOMATIC_TRANSCRIPT_CHARS = 60_000;

function formatProfile(profile: { static: string[]; dynamic: string[] }): string | undefined {
	if (profile.static.length === 0 && profile.dynamic.length === 0) return undefined;
	return prompt.render(profileContextTemplate, {
		static_facts: profile.static.map(item => `- ${escapeSupermemoryXmlText(item)}`).join("\n") || undefined,
		dynamic_facts: profile.dynamic.map(item => `- ${escapeSupermemoryXmlText(item)}`).join("\n") || undefined,
	});
}

function formatSearch(items: SupermemorySearchItem[]): string | undefined {
	if (items.length === 0) return undefined;
	return prompt.render(recallContextTemplate, {
		items: items.map(item => `- ${escapeSupermemoryXmlText(item.content)}`).join("\n"),
	});
}

function activeBranchMessages(session: AgentSession) {
	const branch = session.sessionManager.getBranch();
	const entriesByMessage = new Map<AgentMessage, (typeof branch)[number]>();
	for (const entry of branch) {
		if (entry.type === "message") entriesByMessage.set(entry.message, entry);
	}
	const messages = [];
	for (const displayMessage of session.sessionManager.buildSessionContext({
		transcript: true,
		collapseCompactedHistory: true,
		keepDanglingToolCalls: true,
	}).messages) {
		const entry = entriesByMessage.get(displayMessage);
		if (!entry) continue;
		const message = extractHindsightMessage(entry);
		if (message) messages.push({ ...message, entryId: entry.id });
	}
	return messages;
}
function activeBranchIdentity(session: AgentSession): string {
	const canonicalChildren = new Map<string | null, string>();
	for (const entry of session.sessionManager.getEntries()) {
		if (!canonicalChildren.has(entry.parentId)) canonicalChildren.set(entry.parentId, entry.id);
	}
	const branchChoices: string[] = [];
	for (const entry of session.sessionManager.getBranch()) {
		if (canonicalChildren.get(entry.parentId) !== entry.id) branchChoices.push(entry.id);
	}
	return branchChoices.length > 0 ? branchChoices.join("\u0000") : "trunk";
}

function transcriptForRetentionWindow(
	session: AgentSession,
	firstTurn: number,
	retainEveryNTurns: number,
	forceTail = false,
):
	| { content: string; retainedThroughTurn: number; firstEntryId: string; truncated?: { omittedMessages: number } }
	| undefined {
	const messages = activeBranchMessages(session);
	const userMessageIndexes = messages.flatMap((message, index) => (message.role === "user" ? [index] : []));
	const availableTurns = userMessageIndexes.length;
	const retainedThroughTurn = forceTail ? availableTurns : firstTurn + retainEveryNTurns;
	if (firstTurn < 0 || firstTurn >= availableTurns || (!forceTail && availableTurns < retainedThroughTurn))
		return undefined;
	const start = userMessageIndexes[firstTurn]!;
	const nextWindowStart = userMessageIndexes[retainedThroughTurn] ?? messages.length;
	const entries = messages
		.slice(start, nextWindowStart)
		.map(message => `${message.role === "user" ? "User" : "Assistant"}: ${message.content}`)
		.filter(Boolean);
	const content = entries.join("\n\n").trim();
	if (!content) return undefined;
	if (content.length <= MAX_AUTOMATIC_TRANSCRIPT_CHARS) {
		return { content, retainedThroughTurn, firstEntryId: messages[start]!.entryId };
	}

	const retained: string[] = [];
	for (let index = entries.length - 1; index >= 0; index--) {
		const candidate = [entries[index], ...retained].join("\n\n");
		const header = `[Automatic retention transcript truncated: ${index} earlier message(s) omitted.]\n\n`;
		if (header.length + candidate.length > MAX_AUTOMATIC_TRANSCRIPT_CHARS) break;
		retained.unshift(entries[index]!);
	}
	const omittedMessages = entries.length - retained.length;
	const header = `[Automatic retention transcript truncated: ${omittedMessages} earlier message(s) omitted.]\n\n`;
	const trailing =
		retained.length > 0
			? retained.join("\n\n")
			: entries.at(-1)!.slice(-(MAX_AUTOMATIC_TRANSCRIPT_CHARS - header.length));
	return {
		content: `${header}${trailing}`,
		retainedThroughTurn,
		firstEntryId: messages[start]!.entryId,
		truncated: { omittedMessages },
	};
}

/**
 * Supermemory documents are upserted by `customId`. Each bounded cadence window
 * receives a stable identity based on the durable transcript, scope, first
 * retained turn, first retained entry, and every active choice at a tree branch
 * point. A retry or growing partial tail on the same branch updates that window,
 * while divergent tree branches and later windows cannot overwrite one another.
 *
 * The persisted SessionManager id and entry ids survive resume; provider ids
 * and retention cadence deliberately do not participate in the identity.
 */
function automaticRetentionCustomId(
	session: AgentSession,
	containerTag: string,
	firstTurn: number,
	firstEntryId: string,
	branchIdentity: string,
): string {
	const sessionId = session.sessionManager.getSessionId();
	const digest = new Bun.CryptoHasher("sha256")
		.update(
			`omp-supermemory-retention\u0000${sessionId}\u0000${containerTag}\u0000${firstTurn}\u0000${firstEntryId}\u0000${branchIdentity}`,
		)
		.digest("hex");
	return `omp-retention-${digest}`;
}

function flattenMessagesForQuery(messages: AgentMessage[]): string | undefined {
	for (let index = messages.length - 1; index >= 0; index--) {
		const message = messages[index];
		if (message?.role !== "user") continue;
		if (typeof message.content === "string" && message.content.trim()) return message.content.trim();
		if (Array.isArray(message.content)) {
			const text = message.content
				.filter(
					(block): block is { type: "text"; text: string } =>
						!!block && typeof block === "object" && block.type === "text" && typeof block.text === "string",
				)
				.map(block => block.text)
				.join("\n")
				.trim();
			if (text) return text;
		}
	}
	return undefined;
}

function invalidateLifecycle(state: SupermemorySessionState, preserveActiveRecall = false): number {
	state.pendingRecall = undefined;
	if (!preserveActiveRecall) state.recallInvalidationGeneration += 1;
	state.lifecycleGeneration += 1;
	return state.lifecycleGeneration;
}

function coordinatorKey(config: SupermemoryConfig & { apiKey: string }, containerTag: string): string {
	return new Bun.CryptoHasher("sha256")
		.update(`omp-supermemory-coordinator\u0000${config.baseUrl}\u0000${config.apiKey}\u0000${containerTag}`)
		.digest("hex");
}
function containerState(key: string): SupermemoryContainerState {
	let state = containerStates.get(key);
	if (!state) {
		state = { states: new Set(), documentWrites: new Set(), successfulClearGeneration: 0 };
		containerStates.set(key, state);
	}
	return state;
}

function resetForClear(state: SupermemorySessionState, retainedThroughTurn: number): void {
	invalidateLifecycle(state);
	state.pendingTranscriptResume = false;
	state.lastRecallSnippet = undefined;
	state.hasRecalledForFirstTurn = false;
	state.retention.lastRetainedTurn = retainedThroughTurn;
	state.lastMemoryId = undefined;
	state.lastMemoryStatus = undefined;
	state.lastError = undefined;
}

function clearAdmissionWatermark(state: SupermemorySessionState): ClearAdmissionWatermark {
	return {
		transcriptId: state.session.sessionManager.getSessionId(),
		lifecycleGeneration: state.lifecycleGeneration,
		userTurnCount: activeBranchMessages(state.session).filter(message => message.role === "user").length,
	};
}

function watermarkForSuccessfulClear(state: SupermemorySessionState, admission?: ClearAdmissionWatermark): number {
	if (
		admission &&
		admission.transcriptId === state.session.sessionManager.getSessionId() &&
		admission.lifecycleGeneration === state.lifecycleGeneration
	) {
		return admission.userTurnCount;
	}
	return clearAdmissionWatermark(state).userTurnCount;
}

function admitToActiveClear(state: SupermemorySessionState): void {
	const container = containerStates.get(state.coordinatorKey);
	container?.clearAdmissionWatermarks?.set(state, clearAdmissionWatermark(state));
}

function registerContainerState(state: SupermemorySessionState, containerTag: string): boolean {
	if (state.coordinatorKey) {
		const previous = containerStates.get(state.coordinatorKey);
		previous?.states.delete(state);
		if (
			previous &&
			previous.states.size === 0 &&
			previous.documentWrites.size === 0 &&
			!previous.clearing &&
			previous.successfulClearGeneration === 0
		) {
			containerStates.delete(state.coordinatorKey);
		}
	}
	state.containerTag = containerTag;
	state.coordinatorKey = coordinatorKey(state.config, containerTag);
	const container = containerState(state.coordinatorKey);
	container.states.add(state);
	// A session entering a scope while a clear is in flight must be admitted to
	// that boundary, but its existing watermark is not changed until deletion is
	// confirmed. A failed delete leaves its retention state entirely intact.
	admitToActiveClear(state);
	const observedGeneration = state.observedSuccessfulClearGenerations.get(state.coordinatorKey) ?? 0;
	if (observedGeneration >= container.successfulClearGeneration) return false;
	// This state joined after a completed in-process clear. Its old transcript
	// has already been deleted remotely, so never let automatic retention or a
	// stale recall promote history from before that boundary.
	resetForClear(state, clearAdmissionWatermark(state).userTurnCount);
	state.observedSuccessfulClearGenerations.set(state.coordinatorKey, container.successfulClearGeneration);
	return true;
}

function scopeIsCurrent(state: SupermemorySessionState, scope: SupermemoryScopeSnapshot): boolean {
	return (
		!state.disposed && state.lifecycleGeneration === scope.generation && state.coordinatorKey === scope.coordinatorKey
	);
}

function createTrackedDocument<T>(
	state: SupermemorySessionState,
	scope: SupermemoryScopeSnapshot,
	write: () => Promise<T>,
): Promise<T> | undefined {
	const container = containerState(scope.coordinatorKey);
	if (!scopeIsCurrent(state, scope) || container.clearing) return undefined;
	const promise = write();
	container.documentWrites.add(promise);
	void promise
		.finally(() => {
			container.documentWrites.delete(promise);
			if (
				container.states.size === 0 &&
				container.documentWrites.size === 0 &&
				!container.clearing &&
				container.successfulClearGeneration === 0
			) {
				containerStates.delete(scope.coordinatorKey);
			}
		})
		.catch(() => undefined);
	return promise;
}

function serializeScopeTransition<T>(state: SupermemorySessionState, transition: () => Promise<T>): Promise<T> {
	const prior = state.scopeTransition ?? Promise.resolve();
	const current = prior.catch(() => undefined).then(transition);
	state.scopeTransition = current.then(
		() => undefined,
		() => undefined,
	);
	return current;
}

async function waitForDocumentWrites(container: SupermemoryContainerState): Promise<void> {
	while (container.documentWrites.size > 0) await Promise.allSettled([...container.documentWrites]);
}

async function refreshStateForOperation(
	state: SupermemorySessionState,
	cwd: string,
	session?: AgentSession,
	options?: { refreshPromptContext?: boolean; preserveActiveRecallOnScopeTransition?: boolean },
): Promise<SupermemoryScopeSnapshot | undefined> {
	await state.ready;
	const transition = await serializeScopeTransition(state, async () => {
		if (state.disposed) return { scope: undefined, needsRefresh: false };
		// Credentials and endpoint selection are restart-scoped. Reading the
		// environment again here would let an in-flight session mix old prompt
		// coordination with a new account or transport.
		const refreshed = loadSupermemoryConfig(state.settings);
		const config = { ...refreshed, apiKey: state.config.apiKey, baseUrl: state.config.baseUrl };
		if (!isSupermemoryConfigured(config)) return { scope: undefined, needsRefresh: false };
		state.config = config;
		const containerTag = await resolveSupermemoryContainerTag(cwd, config.scoping);
		if (state.disposed) return { scope: undefined, needsRefresh: false };
		const resumedTranscript = state.pendingTranscriptResume;
		const needsRefresh = containerTag !== state.containerTag;
		if (needsRefresh) {
			invalidateLifecycle(state, options?.preserveActiveRecallOnScopeTransition);
			const joinedAfterClear = registerContainerState(state, containerTag);
			state.retention = {
				scopeGeneration: state.retention.scopeGeneration + 1,
				containerTag,
				// A transcript reset/replay must retain its restored turns from zero.
				// Ordinary cwd movement intentionally isolates existing turns.
				lastRetainedTurn: resumedTranscript
					? 0
					: joinedAfterClear
						? activeBranchMessages(session ?? state.session).filter(message => message.role === "user").length
						: session
							? activeBranchMessages(session).filter(message => message.role === "user").length
							: 0,
			};
			state.hasRecalledForFirstTurn = false;
			state.lastRecallSnippet = undefined;
			state.lastSearchCount = 0;
			state.lastSearchAt = undefined;
		}
		// Reset applies only to the transcript that was just restored. Consume it
		// after the first successful scope refresh even when the tag is unchanged,
		// so a later cwd move cannot replay unrelated accumulated turns from zero.
		if (resumedTranscript) state.pendingTranscriptResume = false;
		// Return an immutable operation scope so a later transition cannot make
		// this operation route through another cwd's client or container.
		return {
			scope: {
				client: state.client,
				config: state.config,
				containerTag: state.containerTag,
				coordinatorKey: state.coordinatorKey,
				generation: state.lifecycleGeneration,
			},
			needsRefresh,
		};
	});
	// Prompt rebuilding calls buildDeveloperInstructions(), which reconciles the
	// scope again. The transition above must settle before that await so the nested
	// no-refresh reconciliation never waits on its own outer refresh.
	if (transition.needsRefresh && options?.refreshPromptContext !== false) {
		try {
			await session?.refreshMemoryPromptContext();
		} catch (error) {
			logger.debug("Supermemory: prompt refresh after scope change failed", { error: String(error) });
		}
	}
	return transition.scope;
}

function createState(
	session: AgentSession,
	settings: Settings,
	automatic: boolean,
): SupermemorySessionState | undefined {
	const config = loadSupermemoryConfig(settings);
	if (!isSupermemoryConfigured(config)) return undefined;
	const state = {
		session,
		client: new SupermemoryClient(config.baseUrl, config.apiKey),
		settings,
		config,
		containerTag: "",
		coordinatorKey: "",
		ready: Promise.resolve(),
		automatic,
		pendingTranscriptResume: false,
		retention: { scopeGeneration: 0, containerTag: "", lastRetainedTurn: 0 },
		lifecycleGeneration: 0,
		recallInvalidationGeneration: 0,
		observedSuccessfulClearGenerations: new Map(),
		hasRecalledForFirstTurn: false,
		lastSearchCount: 0,
		retainPending: false,
		retainForceTail: false,
		retainForceTailEpoch: 0,
		disposed: false,
	} satisfies SupermemorySessionState;
	state.ready = resolveSupermemoryContainerTag(session.sessionManager.getCwd(), config.scoping).then(containerTag => {
		if (state.disposed) return;
		const joinedAfterClear = registerContainerState(state, containerTag);
		// Resume is deliberately not a watermark: a prior process can stop
		// after recording the transcript but before the service acknowledges it.
		// Replaying bounded cadence windows is safe because their customId is
		// stable for this durable transcript and container. A clear completed by
		// this process is different: the remote history is gone.
		state.retention = {
			...state.retention,
			containerTag,
			lastRetainedTurn: joinedAfterClear
				? activeBranchMessages(session).filter(message => message.role === "user").length
				: 0,
		};
	});
	return state;
}

async function saveWithState(
	state: SupermemorySessionState,
	input: MemoryBackendSaveInput,
	cwd: string,
	sessionId?: string,
	retention?: { truncated: boolean; omittedMessages: number },
	session?: AgentSession,
): Promise<MemoryBackendSaveResult> {
	try {
		const scope = await refreshStateForOperation(state, cwd, session);
		if (!scope) return { backend: "supermemory", stored: 0, message: "Supermemory is unavailable or unconfigured." };
		const document = await createTrackedDocument(state, scope, () =>
			scope.client.createDocument({
				content: input.content,
				containerTag: scope.containerTag,
				metadata: {
					source: input.source ?? "omp",
					...(input.context ? { context: input.context } : {}),
					...(sessionId ? { sessionId } : {}),
					...(retention
						? {
								automaticRetention: true,
								transcriptTruncated: retention.truncated,
								omittedMessages: retention.omittedMessages,
							}
						: {}),
				},
			}),
		);
		if (!document) return { backend: "supermemory", stored: 0, message: "Supermemory clear is in progress." };
		if (!scopeIsCurrent(state, scope)) {
			return {
				backend: "supermemory",
				stored: 0,
				message: "Supermemory operation was superseded by a session lifecycle change.",
			};
		}
		state.lastMemoryId = document.id;
		state.lastMemoryStatus = document.status;
		return { backend: "supermemory", stored: 1, ids: [document.id], queued: document.status === "queued" };
	} catch (error) {
		state.lastError = error instanceof Error ? error.message : "Supermemory save failed.";
		logger.warn("Supermemory: save failed", { error: state.lastError });
		return { backend: "supermemory", stored: 0, message: state.lastError };
	}
}

async function searchWithState(
	state: SupermemorySessionState,
	query: string,
	cwd: string,
	options?: MemoryBackendSearchOptions,
	session?: AgentSession,
): Promise<MemoryBackendSearchResult> {
	if (options?.signal?.aborted)
		return { backend: "supermemory", query, count: 0, items: [], message: "Search cancelled." };
	try {
		const scope = await refreshStateForOperation(state, cwd, session);
		if (!scope)
			return {
				backend: "supermemory",
				query,
				count: 0,
				items: [],
				message: "Supermemory is unavailable or unconfigured.",
			};
		const requestedLimit = options?.limit ?? scope.config.recallLimit;
		const normalizedLimit = Number.isFinite(requestedLimit) ? Math.max(0, Math.trunc(requestedLimit)) : 0;
		const resultLimit = Math.min(normalizedLimit, scope.config.recallLimit);
		const response = await scope.client.search({
			q: query,
			containerTag: scope.containerTag,
			searchMode: scope.config.searchMode,
			signal: options?.signal,
			limit: Math.max(2, resultLimit),
			threshold: scope.config.threshold,
		});
		if (options?.signal?.aborted)
			return { backend: "supermemory", query, count: 0, items: [], message: "Search cancelled." };
		if (!scopeIsCurrent(state, scope)) {
			return {
				backend: "supermemory",
				query,
				count: 0,
				items: [],
				message: "Search was superseded by a session lifecycle change.",
			};
		}
		const results = response.results.slice(0, resultLimit);
		state.lastSearchCount = results.length;
		state.lastSearchAt = Date.now();
		return {
			backend: "supermemory",
			query,
			count: results.length,
			items: results.map(item => ({
				id: item.id,
				content: item.content,
				timestamp: item.updatedAt,
				score: item.similarity,
			})),
		};
	} catch (error) {
		state.lastError = error instanceof Error ? error.message : "Supermemory search failed.";
		logger.warn("Supermemory: search failed", { error: state.lastError });
		return { backend: "supermemory", query, count: 0, items: [], message: state.lastError };
	}
}

function preparePendingRecallCommit(
	state: SupermemorySessionState,
	promptText: string,
	options?: MemoryBackendCommitAgentStartOptions,
): MemoryBackendPreparedAgentStartCommit | false | undefined {
	const pending = state.pendingRecall;
	// No preflight ran for this prompt, so this is a no-op rather than an
	// admission rejection. `false` is reserved for a stale, still-staged recall.
	if (!pending) return undefined;
	if (
		pending.promptText !== promptText ||
		pending.generation !== options?.generation ||
		pending.signal !== options?.signal ||
		!recallRequestIsCurrent(state, pending.scope, options)
	) {
		return false;
	}
	return {
		commit() {
			// Agent-core invokes this in the same synchronous admission stack as
			// preparation. Rechecking would turn an accepted prompt into a silent
			// false commit; only consume the exact staged record we prepared.
			if (state.pendingRecall !== pending) return;
			state.pendingRecall = undefined;
			state.hasRecalledForFirstTurn = true;
			state.lastRecallSnippet = pending.snippet;
		},
	};
}

function discardPendingRecall(state: SupermemorySessionState): void {
	state.pendingRecall = undefined;
}

function recallRequestIsCurrent(
	state: SupermemorySessionState,
	scope: SupermemoryScopeSnapshot,
	options?: MemoryBackendBeforeAgentStartOptions,
): boolean {
	return !options?.signal?.aborted && (options?.isCurrent?.() ?? true) && scopeIsCurrent(state, scope);
}

async function recallForFirstTurn(
	state: SupermemorySessionState,
	session: AgentSession,
	promptText: string,
	options?: MemoryBackendBeforeAgentStartOptions,
): Promise<string | undefined> {
	const recallInvalidationGeneration = state.recallInvalidationGeneration;
	discardPendingRecall(state);
	if (options?.signal?.aborted || options?.isCurrent?.() === false) return undefined;
	const scope = await refreshStateForOperation(state, session.sessionManager.getCwd(), session, {
		preserveActiveRecallOnScopeTransition: true,
	});
	if (
		state.recallInvalidationGeneration !== recallInvalidationGeneration ||
		!scope ||
		!state.automatic ||
		!scope.config.autoRecall ||
		state.hasRecalledForFirstTurn ||
		!promptText.trim() ||
		!recallRequestIsCurrent(state, scope, options)
	) {
		return undefined;
	}
	try {
		const [profileResult, searchResult] = await Promise.allSettled([
			scope.client.profile(scope.containerTag, options?.signal),
			scope.client.search({
				q: promptText,
				containerTag: scope.containerTag,
				searchMode: scope.config.searchMode,
				limit: Math.max(2, scope.config.recallLimit),
				threshold: scope.config.threshold,
				signal: options?.signal,
			}),
		]);
		if (
			state.recallInvalidationGeneration !== recallInvalidationGeneration ||
			!recallRequestIsCurrent(state, scope, options)
		)
			return undefined;
		const snippets: string[] = [];
		const errors: string[] = [];
		if (profileResult.status === "fulfilled") {
			const profileSnippet = formatProfile(profileResult.value);
			if (profileSnippet) snippets.push(profileSnippet);
		} else {
			errors.push(
				profileResult.reason instanceof Error ? profileResult.reason.message : "Supermemory profile recall failed.",
			);
		}
		if (searchResult.status === "fulfilled") {
			if (
				state.recallInvalidationGeneration !== recallInvalidationGeneration ||
				!recallRequestIsCurrent(state, scope, options)
			)
				return undefined;
			const results = searchResult.value.results.slice(0, scope.config.recallLimit);
			state.lastSearchCount = results.length;
			state.lastSearchAt = Date.now();
			const searchSnippet = formatSearch(results);
			if (searchSnippet) snippets.push(searchSnippet);
		} else {
			errors.push(
				searchResult.reason instanceof Error ? searchResult.reason.message : "Supermemory search recall failed.",
			);
		}
		if (errors.length > 0) {
			state.lastError = errors.join("; ");
			logger.warn("Supermemory: first-turn recall failed", { error: state.lastError });
		}
		if (
			state.recallInvalidationGeneration !== recallInvalidationGeneration ||
			!recallRequestIsCurrent(state, scope, options)
		)
			return undefined;
		// A completed recall attempt is consumed only when this exact prompt
		// reaches the synchronous agent-core boundary. This includes empty and
		// outage results: otherwise every turn would retry a known first-turn
		// attempt before the agent starts.
		const pending = {
			snippet: snippets.length > 0 ? snippets.join("\n\n") : undefined,
			promptText,
			generation: options?.generation,
			signal: options?.signal,
			scope,
		};
		state.pendingRecall = pending;
		options?.signal?.addEventListener(
			"abort",
			() => {
				if (state.pendingRecall === pending) discardPendingRecall(state);
			},
			{ once: true },
		);
		return pending.snippet;
	} catch (error) {
		if (
			state.recallInvalidationGeneration !== recallInvalidationGeneration ||
			!recallRequestIsCurrent(state, scope, options)
		)
			return undefined;
		state.lastError = error instanceof Error ? error.message : "Supermemory recall failed.";
		logger.warn("Supermemory: first-turn recall failed", { error: state.lastError });
		const pending = { promptText, generation: options?.generation, signal: options?.signal, scope };
		state.pendingRecall = pending;
		options?.signal?.addEventListener(
			"abort",
			() => {
				if (state.pendingRecall === pending) discardPendingRecall(state);
			},
			{ once: true },
		);
		return undefined;
	}
}

async function retainCurrentSession(
	state: SupermemorySessionState,
	session: AgentSession,
	cwd: string,
	forceTail = false,
): Promise<"progressed" | "empty" | "retry" | "stop"> {
	if (state.disposed) return "stop";
	const scope = await refreshStateForOperation(state, cwd, session);
	if (!scope || state.disposed) return "stop";
	if (session.sessionManager.getCwd() !== cwd) return "retry";
	const retention = { ...state.retention };
	const transcript = transcriptForRetentionWindow(
		session,
		retention.lastRetainedTurn,
		scope.config.retainEveryNTurns,
		forceTail,
	);
	if (!transcript) return "empty";
	try {
		const document = await createTrackedDocument(state, scope, () =>
			scope.client.createDocument({
				content: transcript.content,
				containerTag: retention.containerTag,
				customId: automaticRetentionCustomId(
					session,
					retention.containerTag,
					retention.lastRetainedTurn,
					transcript.firstEntryId,
					activeBranchIdentity(session),
				),
				metadata: {
					source: "omp-conversation",
					sessionId: session.sessionId,
					automaticRetention: true,
					...(transcript.truncated
						? { transcriptTruncated: true, omittedMessages: transcript.truncated.omittedMessages }
						: {}),
				},
			}),
		);
		if (!document) {
			// Admission can close in the narrow interval after the loop's clear
			// check. Keep this request for the clear boundary to replay rather
			// than silently dropping the unretained transcript window.
			if (containerState(scope.coordinatorKey).clearing) state.retainPending = true;
			return "stop";
		}
		if (
			scopeIsCurrent(state, scope) &&
			state.retention.scopeGeneration === retention.scopeGeneration &&
			state.retention.containerTag === retention.containerTag &&
			state.retention.lastRetainedTurn === retention.lastRetainedTurn
		) {
			state.retention.lastRetainedTurn = transcript.retainedThroughTurn;
			state.lastMemoryId = document.id;
			state.lastMemoryStatus = document.status;
			return "progressed";
		}
		return "retry";
	} catch (error) {
		if (scopeIsCurrent(state, scope)) {
			state.lastError = error instanceof Error ? error.message : "Supermemory retention failed.";
			logger.warn("Supermemory: retention failed", { error: state.lastError });
		}
		return "stop";
	}
}

function requestAutomaticRetention(state: SupermemorySessionState, session: AgentSession, forceTail = false): void {
	if (state.disposed) return;
	state.retainPending = true;
	if (forceTail) {
		state.retainForceTail = true;
		state.retainForceTailEpoch++;
	}
	if (state.retainInFlight) return;
	const retain = (async () => {
		while (!state.disposed) {
			state.retainPending = false;
			const retainTail = state.retainForceTail;
			const forceTailEpoch = state.retainForceTailEpoch;
			if (!state.automatic) return;
			const config = loadSupermemoryConfig(state.settings);
			// A clear closes document admission. Preserve the pending request for
			// clear's finally block, which re-schedules live states after either
			// outcome (including autoRetain=false).
			if (containerState(state.coordinatorKey).clearing) {
				state.retainPending = true;
				return;
			}
			if (!isSupermemoryConfigured(config) || (!config.autoRetain && !retainTail)) return;
			const outcome = await retainCurrentSession(state, session, session.sessionManager.getCwd(), retainTail);
			// A forced tail is consumed only by a successful document write. An
			// empty transcript is also terminally confirmed. Preserve it after
			// failures/retries so a later clear boundary or explicit enqueue can
			// retry it.
			if (
				retainTail &&
				state.retainForceTailEpoch === forceTailEpoch &&
				(outcome === "progressed" || outcome === "empty")
			) {
				state.retainForceTail = false;
			}
			if ((outcome === "stop" || outcome === "empty") && !state.retainPending) return;
		}
	})().finally(() => {
		state.retainInFlight = undefined;
		if (state.retainPending && !state.disposed && !containerState(state.coordinatorKey).clearing) {
			requestAutomaticRetention(state, session);
		}
	});
	state.retainInFlight = retain;
	void retain;
}

async function waitForRetention(state: SupermemorySessionState, bounded = false): Promise<void> {
	if (!state.retainInFlight) return;
	if (!bounded) {
		await state.retainInFlight.catch(() => undefined);
		return;
	}
	const timeout = Promise.withResolvers<void>();
	const timer = setTimeout(timeout.resolve, RETENTION_SHUTDOWN_TIMEOUT_MS);
	try {
		await Promise.race([state.retainInFlight.catch(() => undefined), timeout.promise]);
	} finally {
		clearTimeout(timer);
	}
}

async function flushRetentionTail(
	state: SupermemorySessionState,
	session: AgentSession,
	cwd: string,
	explicit = false,
): Promise<void> {
	if (!state.automatic || state.disposed) return;
	const scope = await refreshStateForOperation(state, cwd, session);
	if (!scope || state.disposed || (!explicit && !scope.config.autoRetain)) return;
	requestAutomaticRetention(state, session, true);
	for (;;) {
		const clearing = containerState(scope.coordinatorKey).clearing;
		if (!clearing) break;
		await clearing.catch(() => undefined);
	}
	await waitForRetention(state);
	if (state.retainForceTail) throw new Error(state.lastError ?? "Supermemory retention failed.");
}

function statusForState(state: SupermemorySessionState): MemoryBackendStatus {
	return {
		backend: "supermemory",
		active: true,
		writable: true,
		searchable: true,
		scope: state.containerTag,
		lastMemory: state.lastMemoryId,
		lastRecall: state.hasRecalledForFirstTurn,
		message: state.lastMemoryStatus ? `Last document status: ${state.lastMemoryStatus}` : undefined,
		error: state.lastError,
	};
}

export const supermemoryBackend: MemoryBackend &
	Required<
		Pick<
			MemoryBackend,
			| "status"
			| "search"
			| "save"
			| "stats"
			| "diagnose"
			| "beforeAgentStartPrompt"
			| "commitBeforeAgentStartPrompt"
			| "preCompactionContext"
			| "beforeTranscriptReplace"
			| "resetSession"
			| "disposeSession"
		>
	> = {
	id: "supermemory",

	async start(options: MemoryBackendStartOptions): Promise<void> {
		const { session } = options;
		const prior = sessionStates.get(session);
		prior?.unsubscribe?.();
		sessionStates.delete(session);
		const state = createState(session, options.settings, session.agentKind === "main");
		if (!state) {
			logger.warn("Supermemory: memory.backend=supermemory but SUPERMEMORY_API_KEY is unavailable; backend inert.");
			return;
		}
		sessionStates.set(session, state);
		try {
			await state.ready;
		} catch (error) {
			sessionStates.delete(session);
			logger.warn("Supermemory: backend initialization failed", {
				error: error instanceof Error ? error.message : "Unknown initialization failure.",
			});
			return;
		}
		// Disposal or a newer start can happen while container resolution is
		// pending. Never attach a listener for a stale state.
		if (state.disposed || sessionStates.get(session) !== state) return;
		if (!state.automatic) return;
		state.unsubscribe = session.subscribe(event => {
			if (event.type === "agent_end") requestAutomaticRetention(state, session);
		});
	},

	async buildDeveloperInstructions(_agentDir, settings, session): Promise<string | undefined> {
		const state = session && sessionStates.get(session);
		if (!state && !isSupermemoryConfigured(loadSupermemoryConfig(settings))) return undefined;
		// Prompt rebuilding can be the first operation after `/move`. Reconcile
		// the session's live cwd here, but never invoke the prompt-refresh hook
		// from inside prompt assembly.
		if (state && session) {
			await refreshStateForOperation(state, session.sessionManager.getCwd(), session, {
				refreshPromptContext: false,
			});
		}
		return [instructions.trim(), state?.lastRecallSnippet].filter((value): value is string => !!value).join("\n\n");
	},

	async beforeAgentStartPrompt(
		session: AgentSession,
		promptText: string,
		options?: MemoryBackendBeforeAgentStartOptions,
	): Promise<string | undefined> {
		const state = sessionStates.get(session);
		return state ? await recallForFirstTurn(state, session, promptText, options) : undefined;
	},

	async commitBeforeAgentStartPrompt(
		session: AgentSession,
		promptText: string,
		options?: MemoryBackendCommitAgentStartOptions,
	): Promise<MemoryBackendPreparedAgentStartCommit | false | undefined> {
		const state = sessionStates.get(session);
		return state ? preparePendingRecallCommit(state, promptText, options) : undefined;
	},

	async clear(_agentDir, cwd, session): Promise<void> {
		const state = session && sessionStates.get(session);
		if (!state || !session)
			throw new Error("Supermemory is unavailable: no configured active session state to clear.");
		const scope = await refreshStateForOperation(state, cwd, session);
		if (!scope) throw new Error("Supermemory is unavailable or unconfigured.");
		const container = containerState(scope.coordinatorKey);
		// Each delete owns its admission map only while it is the executing
		// boundary. Queued deletes must not steal admissions from the active one.
		let activeClearWatermarks: Map<SupermemorySessionState, ClearAdmissionWatermark> | undefined;
		const clear = async () => {
			const clearWatermarks = new Map(
				[...container.states]
					.filter(liveState => liveState.containerTag === scope.containerTag)
					.map(liveState => [liveState, clearAdmissionWatermark(liveState)]),
			);
			activeClearWatermarks = clearWatermarks;
			container.clearAdmissionWatermarks = clearWatermarks;
			await Promise.all([...container.states].map(liveState => waitForRetention(liveState)));
			await waitForDocumentWrites(container);
			const deleted = await scope.client.deleteContainerTag(scope.containerTag);
			if (!deleted.success || deleted.containerTag !== scope.containerTag) {
				throw new Error("Supermemory did not confirm deletion of the requested memory container.");
			}
			// Admissions can arrive while the delete or an earlier peer refresh is
			// awaiting. Keep draining the staged records so each admitted state is
			// reset at its own boundary and has its prompt refreshed before clear
			// resolves.
			const refreshedAdmissions = new Map<SupermemorySessionState, ClearAdmissionWatermark>();
			for (;;) {
				const pendingAdmissions = [...clearWatermarks].flatMap(([liveState, admission]) => {
					if (
						refreshedAdmissions.get(liveState) === admission ||
						liveState.disposed ||
						liveState.containerTag !== scope.containerTag ||
						sessionStates.get(liveState.session) !== liveState
					) {
						return [];
					}
					return [
						{ liveState, admission, retainedThroughTurn: watermarkForSuccessfulClear(liveState, admission) },
					];
				});
				if (pendingAdmissions.length === 0) break;

				for (const { liveState, retainedThroughTurn } of pendingAdmissions) {
					resetForClear(liveState, retainedThroughTurn);
				}
				await Promise.all(
					pendingAdmissions.map(async ({ liveState, admission }) => {
						try {
							await liveState.session.refreshMemoryPromptContext();
						} catch (error) {
							logger.debug("Supermemory: prompt refresh after clear failed", { error: String(error) });
						} finally {
							if (clearWatermarks.get(liveState) === admission) refreshedAdmissions.set(liveState, admission);
						}
					}),
				);
			}
			// Seal this admission map before resolving the clear. A state that
			// registers after this point observes the completed generation below
			// rather than joining an already-drained map.
			container.successfulClearGeneration += 1;
			for (const [liveState] of clearWatermarks) {
				liveState.observedSuccessfulClearGenerations.set(scope.coordinatorKey, container.successfulClearGeneration);
			}
			if (container.clearAdmissionWatermarks === clearWatermarks) container.clearAdmissionWatermarks = undefined;
		};
		const previousClear = container.clearing;
		const task = (previousClear ? previousClear.catch(() => undefined) : Promise.resolve()).then(clear);
		container.clearing = task;
		void task.then(
			() => {
				if (container.clearAdmissionWatermarks === activeClearWatermarks)
					container.clearAdmissionWatermarks = undefined;
			},
			() => {
				if (container.clearAdmissionWatermarks === activeClearWatermarks)
					container.clearAdmissionWatermarks = undefined;
			},
		);
		try {
			await task;
		} finally {
			if (container.clearing === task) {
				container.clearing = undefined;
				container.clearAdmissionWatermarks = undefined;
				// A peer can finish a complete retention window while this clear is
				// draining. Its first scheduling pass exits while `clearing` is set,
				// so re-evaluate every state that is still live in this coordinator
				// after the boundary opens, whether deletion succeeded or failed.
				for (const liveState of container.states) {
					if (
						liveState.disposed ||
						liveState.containerTag !== scope.containerTag ||
						sessionStates.get(liveState.session) !== liveState
					) {
						continue;
					}
					if (!liveState.retainPending && !liveState.retainForceTail) continue;
					requestAutomaticRetention(liveState, liveState.session);
				}
			}
		}
	},

	async enqueue(_agentDir, cwd, session): Promise<void> {
		const state = session && sessionStates.get(session);
		if (!state || !session) return;
		await flushRetentionTail(state, session, cwd || session.sessionManager.getCwd(), true);
	},

	async status(context: MemoryBackendOperationContext): Promise<MemoryBackendStatus> {
		const state = context.session && sessionStates.get(context.session);
		if (!state)
			return {
				backend: "supermemory",
				active: false,
				writable: false,
				searchable: false,
				message: "No active Supermemory session state.",
			};
		await refreshStateForOperation(state, context.cwd, context.session);
		return statusForState(state);
	},

	async search(
		context: MemoryBackendOperationContext,
		query: string,
		options?: MemoryBackendSearchOptions,
	): Promise<MemoryBackendSearchResult> {
		const state = context.session && sessionStates.get(context.session);
		return state
			? await searchWithState(state, query, context.cwd, options, context.session)
			: { backend: "supermemory", query, count: 0, items: [], message: "No active Supermemory session state." };
	},

	async save(context: MemoryBackendOperationContext, input: MemoryBackendSaveInput): Promise<MemoryBackendSaveResult> {
		const state = context.session && sessionStates.get(context.session);
		return state
			? await saveWithState(state, input, context.cwd, context.session?.sessionId, undefined, context.session)
			: { backend: "supermemory", stored: 0, message: "No active Supermemory session state." };
	},

	async preCompactionContext(
		messages: AgentMessage[],
		_settings: Settings,
		session?: AgentSession,
	): Promise<string | undefined> {
		const state = session && sessionStates.get(session);
		if (!state || !session || !state.automatic) return undefined;
		const query = flattenMessagesForQuery(messages);
		if (!query) return undefined;
		const result = await searchWithState(state, query, session.sessionManager.getCwd(), undefined, session);
		return formatSearch(
			result.items.map(item => ({
				id: item.id ?? "",
				content: item.content,
				similarity: item.score,
				updatedAt: item.timestamp,
			})),
		);
	},

	async stats(_agentDir, cwd, session): Promise<string | undefined> {
		const state = session && sessionStates.get(session);
		if (!state) return undefined;
		await refreshStateForOperation(state, cwd, session);
		return `## Supermemory\n\n- Scope: \`${state.containerTag}\`\n- Last search results: ${state.lastSearchCount}\n- Last document: ${state.lastMemoryId ?? "none"} (${state.lastMemoryStatus ?? "n/a"})`;
	},

	async diagnose(_agentDir, cwd, session): Promise<string | undefined> {
		const state = session && sessionStates.get(session);
		if (!state)
			return "## Supermemory\n\nNo active session state. Set `SUPERMEMORY_API_KEY` and restart the session.";
		await refreshStateForOperation(state, cwd, session);
		return `## Supermemory\n\n- API key: configured (not displayed)\n- Scope: \`${state.containerTag}\`\n- Automatic recall/retention: ${state.automatic ? "primary session only" : "disabled for this subagent"}\n- Last request error: ${state.lastError ?? "none"}`;
	},

	async beforeTranscriptReplace(session: AgentSession): Promise<void> {
		const state = sessionStates.get(session);
		if (!state) return;
		await flushRetentionTail(state, session, session.sessionManager.getCwd());
	},

	resetSession(session: AgentSession): boolean {
		const state = sessionStates.get(session);
		if (!state) return false;
		invalidateLifecycle(state);
		admitToActiveClear(state);
		state.pendingTranscriptResume = true;
		state.retention.lastRetainedTurn = 0;
		state.hasRecalledForFirstTurn = false;
		state.lastRecallSnippet = undefined;
		return true;
	},

	async disposeSession(session: AgentSession): Promise<void> {
		const state = sessionStates.get(session);
		if (!state) return;
		invalidateLifecycle(state);
		state.disposed = true;
		state.unsubscribe?.();
		await waitForRetention(state, true);
		if (state.coordinatorKey) {
			const container = containerStates.get(state.coordinatorKey);
			container?.states.delete(state);
			if (
				container &&
				container.states.size === 0 &&
				container.documentWrites.size === 0 &&
				!container.clearing &&
				container.successfulClearGeneration === 0
			) {
				containerStates.delete(state.coordinatorKey);
			}
		}
		sessionStates.delete(session);
	},
};
