import type { SupermemorySearchMode } from "./config";

export interface SupermemoryDocument {
	id: string;
	status: string;
}

export interface SupermemoryDeletedContainerTag {
	success: boolean;
	containerTag: string;
	deletedDocumentsCount: number;
	deletedMemoriesCount: number;
}

export interface SupermemorySearchItem {
	id: string;
	content: string;
	similarity?: number;
	updatedAt?: string;
	metadata?: Record<string, unknown> | null;
}

export interface SupermemoryProfile {
	static: string[];
	dynamic: string[];
}

interface SupermemorySearchResponse {
	results: SupermemorySearchItem[];
	total: number;
	timing?: number;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
	return value !== null && typeof value === "object" && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: undefined;
}

function asStringArray(value: unknown): string[] {
	return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function parseDocument(value: unknown): SupermemoryDocument {
	const object = asRecord(value);
	if (!object || typeof object.id !== "string" || typeof object.status !== "string") {
		throw new Error("Supermemory returned an invalid document response.");
	}
	return { id: object.id, status: object.status };
}

function parseDeletedContainerTag(value: unknown): SupermemoryDeletedContainerTag {
	const object = asRecord(value);
	if (
		!object ||
		typeof object.success !== "boolean" ||
		typeof object.containerTag !== "string" ||
		typeof object.deletedDocumentsCount !== "number" ||
		typeof object.deletedMemoriesCount !== "number"
	) {
		throw new Error("Supermemory returned an invalid container deletion response.");
	}
	if (!object.success) throw new Error("Supermemory returned an unsuccessful clear response.");
	return {
		success: object.success,
		containerTag: object.containerTag,
		deletedDocumentsCount: object.deletedDocumentsCount,
		deletedMemoriesCount: object.deletedMemoriesCount,
	};
}

function parseSearch(value: unknown): SupermemorySearchResponse {
	const object = asRecord(value);
	const rawResults = object?.results;
	if (!object || !Array.isArray(rawResults)) throw new Error("Supermemory returned an invalid search response.");
	const results: SupermemorySearchItem[] = [];
	for (const raw of rawResults) {
		const item = asRecord(raw);
		if (!item || typeof item.id !== "string") continue;
		const content =
			typeof item.memory === "string" ? item.memory : typeof item.chunk === "string" ? item.chunk : undefined;
		if (!content) continue;
		results.push({
			id: item.id,
			content,
			similarity: typeof item.similarity === "number" ? item.similarity : undefined,
			updatedAt: typeof item.updatedAt === "string" ? item.updatedAt : undefined,
			metadata: asRecord(item.metadata) ?? null,
		});
	}
	return {
		results,
		total: typeof object.total === "number" ? object.total : results.length,
		timing: typeof object.timing === "number" ? object.timing : undefined,
	};
}

function parseProfile(value: unknown): SupermemoryProfile {
	const root = asRecord(value);
	const profile = root && asRecord(root.profile);
	if (!profile) throw new Error("Supermemory returned an invalid profile response.");
	return { static: asStringArray(profile.static), dynamic: asStringArray(profile.dynamic) };
}

const DEFAULT_REQUEST_TIMEOUT_MS = 15_000;

/** Thin wrapper around the documented v3 documents and v4 recall/profile endpoints. */
export class SupermemoryClient {
	readonly #baseUrl: string;
	readonly #apiKey: string;
	readonly #requestTimeoutMs: number;

	constructor(baseUrl: string, apiKey: string, requestTimeoutMs = DEFAULT_REQUEST_TIMEOUT_MS) {
		this.#baseUrl = baseUrl.replace(/\/+$/, "");
		this.#apiKey = apiKey;
		this.#requestTimeoutMs = Number.isFinite(requestTimeoutMs)
			? Math.min(60_000, Math.max(1, Math.floor(requestTimeoutMs)))
			: DEFAULT_REQUEST_TIMEOUT_MS;
	}

	async createDocument(input: {
		content: string;
		containerTag: string;
		customId?: string;
		metadata?: Record<string, string | number | boolean>;
	}): Promise<SupermemoryDocument> {
		return parseDocument(await this.#request("/v3/documents", "POST", input));
	}

	async search(input: {
		q: string;
		containerTag: string;
		searchMode: SupermemorySearchMode;
		limit: number;
		threshold: number;
		signal?: AbortSignal;
	}): Promise<SupermemorySearchResponse> {
		const { signal, ...body } = input;
		return parseSearch(await this.#request("/v4/search", "POST", body, signal));
	}

	async profile(containerTag: string, signal?: AbortSignal): Promise<SupermemoryProfile> {
		return parseProfile(await this.#request("/v4/profile", "POST", { containerTag }, signal));
	}

	async deleteContainerTag(containerTag: string): Promise<SupermemoryDeletedContainerTag> {
		return parseDeletedContainerTag(
			await this.#request(`/v3/container-tags/${encodeURIComponent(containerTag)}`, "DELETE"),
		);
	}

	async #request(
		pathname: string,
		method: "POST" | "DELETE",
		body?: unknown,
		callerSignal?: AbortSignal,
	): Promise<unknown> {
		if (callerSignal?.aborted) throw new Error("Supermemory request cancelled.");
		const timeoutSignal = AbortSignal.timeout(this.#requestTimeoutMs);
		const signal = callerSignal ? AbortSignal.any([callerSignal, timeoutSignal]) : timeoutSignal;
		let response: Response;
		try {
			response = await fetch(`${this.#baseUrl}${pathname}`, {
				method,
				headers: { Authorization: `Bearer ${this.#apiKey}`, "Content-Type": "application/json" },
				body: body === undefined ? undefined : JSON.stringify(body),
				signal,
			});
		} catch {
			if (callerSignal?.aborted) throw new Error("Supermemory request cancelled.");
			if (timeoutSignal.aborted) throw new Error("Supermemory request timed out.");
			throw new Error("Supermemory request failed.");
		}
		if (!response.ok) throw new Error(`Supermemory request failed with HTTP ${response.status}.`);
		try {
			return await response.json();
		} catch {
			throw new Error("Supermemory returned invalid JSON.");
		}
	}
}
