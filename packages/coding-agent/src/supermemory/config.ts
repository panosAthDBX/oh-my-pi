import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { Settings } from "../config/settings";
import * as git from "../utils/git";

export type SupermemoryScoping = "global" | "per-project";
export type SupermemorySearchMode = "hybrid" | "memories";

export interface SupermemoryConfig {
	baseUrl: string;
	scoping: SupermemoryScoping;
	autoRecall: boolean;
	autoRetain: boolean;
	retainEveryNTurns: number;
	recallLimit: number;
	threshold: number;
	searchMode: SupermemorySearchMode;
	apiKey: string | null;
}

const DEFAULT_BASE_URL = "https://api.supermemory.ai";
const DEFAULT_RETAIN_EVERY_N_TURNS = 3;
const DEFAULT_RECALL_LIMIT = 8;
const DEFAULT_THRESHOLD = 0.5;

function nonEmptyString(value: unknown): string | undefined {
	return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function finiteNumber(value: unknown, fallback: number, minimum: number, maximum: number): number {
	const numeric =
		typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : Number.NaN;
	return Number.isFinite(numeric) ? Math.min(maximum, Math.max(minimum, numeric)) : fallback;
}

function pickScoping(value: unknown): SupermemoryScoping {
	return value === "global" || value === "per-project" ? value : "per-project";
}

function pickSearchMode(value: unknown): SupermemorySearchMode {
	return value === "memories" || value === "hybrid" ? value : "hybrid";
}

function resolveBaseUrl(env: NodeJS.ProcessEnv): string | undefined {
	const override = nonEmptyString(env.SUPERMEMORY_BASE_URL);
	if (!override) return DEFAULT_BASE_URL;
	try {
		const url = new URL(override);
		const isLoopback =
			url.hostname === "localhost" ||
			url.hostname.endsWith(".localhost") ||
			url.hostname === "127.0.0.1" ||
			url.hostname === "[::1]";
		return url.protocol === "https:" || (url.protocol === "http:" && isLoopback)
			? url.toString().replace(/\/+$/, "")
			: undefined;
	} catch {
		return undefined;
	}
}

/** Resolves settings plus process-only API origin and credential without persisting either secret-bearing value. */
export function loadSupermemoryConfig(settings: Settings, env: NodeJS.ProcessEnv = process.env): SupermemoryConfig {
	const baseUrl = resolveBaseUrl(env);
	return {
		baseUrl: baseUrl ?? "",
		scoping: pickScoping(settings.get("supermemory.scoping")),
		autoRecall: settings.get("supermemory.autoRecall") ?? true,
		autoRetain: settings.get("supermemory.autoRetain") ?? true,
		retainEveryNTurns: Math.floor(
			finiteNumber(settings.get("supermemory.retainEveryNTurns"), DEFAULT_RETAIN_EVERY_N_TURNS, 1, 100),
		),
		recallLimit: Math.floor(finiteNumber(settings.get("supermemory.recallLimit"), DEFAULT_RECALL_LIMIT, 1, 50)),
		threshold: finiteNumber(settings.get("supermemory.threshold"), DEFAULT_THRESHOLD, 0, 1),
		searchMode: pickSearchMode(settings.get("supermemory.searchMode")),
		apiKey: baseUrl ? (nonEmptyString(env.SUPERMEMORY_API_KEY) ?? null) : null,
	};
}

export function isSupermemoryConfigured(config: SupermemoryConfig): config is SupermemoryConfig & { apiKey: string } {
	return config.apiKey !== null && config.baseUrl !== "";
}

/**
 * Returns a stable opaque container tag. Repositories use OMP's shared primary
 * repository identity, so a subdirectory and every linked worktree share one
 * scope. Outside a repository we fall back to the physical directory path.
 * The digest prevents Supermemory from receiving either local path.
 */
export async function resolveSupermemoryContainerTag(cwd: string, scoping: SupermemoryScoping): Promise<string> {
	if (scoping === "global") return "omp-global";
	const repositoryIdentity = git.repo.primaryRootSync(cwd);
	let canonicalIdentity = path.resolve(repositoryIdentity ?? cwd);
	try {
		canonicalIdentity = await fs.realpath(canonicalIdentity);
	} catch {
		// A directory may disappear during session teardown; the resolved path
		// still provides a deterministic, non-secret fallback scope.
	}
	const normalized = canonicalIdentity.replace(/\\/g, "/");
	const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(normalized));
	const hex = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
	return `omp-project-${hex.slice(0, 24)}`;
}
