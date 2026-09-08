import { afterEach, describe, expect, it, vi } from "bun:test";
import * as path from "node:path";
import { SupermemoryClient } from "../src/supermemory/client";
import {
	isSupermemoryConfigured,
	loadSupermemoryConfig,
	resolveSupermemoryContainerTag,
} from "../src/supermemory/config";

function fakeSettings(values: Record<string, unknown> = {}) {
	return { get: (key: string) => values[key] } as never;
}

describe("Supermemory configuration", () => {
	it("uses safe validated defaults, ignores project-configured origins, and reads credential/origin only from its supplied environment", () => {
		const config = loadSupermemoryConfig(
			fakeSettings({
				"supermemory.baseUrl": "https://credential-exfiltration.example",
				"supermemory.scoping": "invalid",
				"supermemory.retainEveryNTurns": 0,
				"supermemory.recallLimit": 99,
				"supermemory.threshold": 2,
				"supermemory.searchMode": "invalid",
			}),
			{ SUPERMEMORY_API_KEY: "  test-secret  " },
		);

		expect(config).toEqual({
			baseUrl: "https://api.supermemory.ai",
			scoping: "per-project",
			autoRecall: true,
			autoRetain: true,
			retainEveryNTurns: 1,
			recallLimit: 50,
			threshold: 1,
			searchMode: "hybrid",
			apiKey: "test-secret",
		});
		expect(isSupermemoryConfigured(config)).toBe(true);
		expect(isSupermemoryConfigured(loadSupermemoryConfig(fakeSettings(), {}))).toBe(false);
	});

	it("coerces numeric submenu string values without accepting nonnumeric settings", () => {
		const config = loadSupermemoryConfig(
			fakeSettings({
				"supermemory.retainEveryNTurns": "5",
				"supermemory.recallLimit": "12",
				"supermemory.threshold": "0.8",
			}),
			{ SUPERMEMORY_API_KEY: "test-secret" },
		);
		expect(config).toMatchObject({ retainEveryNTurns: 5, recallLimit: 12, threshold: 0.8 });
		expect(
			loadSupermemoryConfig(fakeSettings({ "supermemory.recallLimit": "twelve" }), {
				SUPERMEMORY_API_KEY: "test-secret",
			}).recallLimit,
		).toBe(8);
		expect(
			loadSupermemoryConfig(fakeSettings({ "supermemory.recallLimit": 1 }), {
				SUPERMEMORY_API_KEY: "test-secret",
			}).recallLimit,
		).toBe(1);
	});

	it("allows HTTPS and loopback HTTP process-only origin overrides but rejects remote plaintext origins", () => {
		expect(
			loadSupermemoryConfig(fakeSettings(), {
				SUPERMEMORY_API_KEY: "secret",
				SUPERMEMORY_BASE_URL: "https://memory.example/",
			}).baseUrl,
		).toBe("https://memory.example");
		expect(
			loadSupermemoryConfig(fakeSettings(), {
				SUPERMEMORY_API_KEY: "secret",
				SUPERMEMORY_BASE_URL: "http://localhost:6767/",
			}).baseUrl,
		).toBe("http://localhost:6767");
		for (const rejectedOrigin of ["http://memory.example", "http://127.evil.com", "not a URL"]) {
			const config = loadSupermemoryConfig(fakeSettings(), {
				SUPERMEMORY_API_KEY: "secret",
				SUPERMEMORY_BASE_URL: rejectedOrigin,
			});
			expect(config.baseUrl).toBe("");
			expect(config.apiKey).toBeNull();
			expect(isSupermemoryConfigured(config)).toBe(false);
		}
	});

	it("uses stable opaque project tags and a stable global tag", async () => {
		const first = await resolveSupermemoryContainerTag("/tmp/project", "per-project");
		const second = await resolveSupermemoryContainerTag("/tmp/project", "per-project");
		expect(first).toBe(second);
		expect(first).toMatch(/^omp-project-[a-f0-9]{24}$/);
		expect(first).not.toContain("/tmp/project");
		expect(await resolveSupermemoryContainerTag("/elsewhere", "global")).toBe("omp-global");
	});

	it("uses the common repository identity for a repository root and its subdirectories", async () => {
		const packageRoot = path.resolve(import.meta.dir, "..");
		const nestedSourceDirectory = path.join(packageRoot, "src");
		expect(await resolveSupermemoryContainerTag(packageRoot, "per-project")).toBe(
			await resolveSupermemoryContainerTag(nestedSourceDirectory, "per-project"),
		);
	});
});

describe("SupermemoryClient", () => {
	afterEach(() => vi.restoreAllMocks());

	it("maps documented endpoints, auth, and request payloads exactly", async () => {
		const fetchMock = vi
			.spyOn(globalThis, "fetch")
			.mockResolvedValueOnce(new Response(JSON.stringify({ id: "doc-1", status: "queued" }), { status: 200 }))
			.mockResolvedValueOnce(
				new Response(
					JSON.stringify({ results: [{ id: "m-1", memory: "remembered", similarity: 0.9 }], total: 1 }),
					{
						status: 200,
					},
				),
			)
			.mockResolvedValueOnce(
				new Response(JSON.stringify({ profile: { static: ["prefers terse"], dynamic: [] } }), { status: 200 }),
			);
		const client = new SupermemoryClient("https://example.test/", "test-secret");

		await client.createDocument({
			content: "save this",
			containerTag: "omp-global",
			customId: "omp-retention-opaque-document-id",
			metadata: { source: "test" },
		});
		await client.search({ q: "what", containerTag: "omp-global", searchMode: "hybrid", limit: 3, threshold: 0.4 });
		await client.profile("omp-global");

		expect(fetchMock).toHaveBeenCalledTimes(3);
		const [documentUrl, documentInit] = fetchMock.mock.calls[0]!;
		expect(documentUrl).toBe("https://example.test/v3/documents");
		expect(documentInit).toMatchObject({
			method: "POST",
			headers: { Authorization: "Bearer test-secret", "Content-Type": "application/json" },
		});
		expect(JSON.parse(documentInit!.body as string)).toEqual({
			content: "save this",
			containerTag: "omp-global",
			customId: "omp-retention-opaque-document-id",
			metadata: { source: "test" },
		});
		expect(fetchMock.mock.calls.slice(1).map(([url]) => url)).toEqual([
			"https://example.test/v4/search",
			"https://example.test/v4/profile",
		]);
		expect(JSON.parse(fetchMock.mock.calls[1]![1]!.body as string)).toEqual({
			q: "what",
			containerTag: "omp-global",
			searchMode: "hybrid",
			limit: 3,
			threshold: 0.4,
		});
	});

	it("deletes an entire scoped container only when the API confirms success", async () => {
		const fetchMock = vi
			.spyOn(globalThis, "fetch")
			.mockResolvedValueOnce(
				new Response(
					JSON.stringify({
						success: true,
						containerTag: "omp-global",
						deletedDocumentsCount: 2,
						deletedMemoriesCount: 3,
					}),
					{
						status: 200,
					},
				),
			)
			.mockResolvedValueOnce(
				new Response(
					JSON.stringify({
						success: false,
						containerTag: "omp-global",
						deletedDocumentsCount: 0,
						deletedMemoriesCount: 0,
					}),
					{ status: 200 },
				),
			);
		const client = new SupermemoryClient("https://example.test", "test-secret");

		await expect(client.deleteContainerTag("omp-global")).resolves.toEqual({
			success: true,
			containerTag: "omp-global",
			deletedDocumentsCount: 2,
			deletedMemoriesCount: 3,
		});
		await expect(client.deleteContainerTag("omp-global")).rejects.toThrow(
			"Supermemory returned an unsuccessful clear response.",
		);
		expect(fetchMock.mock.calls[0]).toMatchObject([
			"https://example.test/v3/container-tags/omp-global",
			{ method: "DELETE", headers: { Authorization: "Bearer test-secret" } },
		]);
	});

	it("redacts transport failures without exposing an API credential", async () => {
		vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Authorization: Bearer test-secret"));
		const client = new SupermemoryClient("https://example.test", "test-secret");
		await expect(
			client.search({ q: "x", containerTag: "scope", searchMode: "hybrid", limit: 1, threshold: 0 }),
		).rejects.toThrow("Supermemory request failed.");
	});

	it("bounds a hung request with the client timeout", async () => {
		vi.spyOn(globalThis, "fetch").mockImplementation(
			Object.assign(
				(...[_url, init]: Parameters<typeof fetch>) => {
					const { promise, reject } = Promise.withResolvers<Response>();
					const signal = (init as RequestInit).signal;
					signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
					return promise;
				},
				{ preconnect: globalThis.fetch.preconnect },
			),
		);
		const client = new SupermemoryClient("https://example.test", "test-secret", 1);

		await expect(
			client.search({ q: "x", containerTag: "scope", searchMode: "hybrid", limit: 1, threshold: 0 }),
		).rejects.toThrow("Supermemory request timed out.");
	});

	it("composes a caller abort signal into search requests", async () => {
		const controller = new AbortController();
		const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
			Object.assign(
				(...[_url, init]: Parameters<typeof fetch>) => {
					const { promise, reject } = Promise.withResolvers<Response>();
					const signal = (init as RequestInit).signal;
					signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
					return promise;
				},
				{ preconnect: globalThis.fetch.preconnect },
			),
		);
		const client = new SupermemoryClient("https://example.test", "test-secret", 1_000);
		const search = client.search({
			q: "x",
			containerTag: "scope",
			searchMode: "hybrid",
			limit: 1,
			threshold: 0,
			signal: controller.signal,
		});
		controller.abort();

		await expect(search).rejects.toThrow("Supermemory request cancelled.");
		expect((fetchMock.mock.calls[0]![1] as RequestInit).signal).not.toBe(controller.signal);
	});
});
