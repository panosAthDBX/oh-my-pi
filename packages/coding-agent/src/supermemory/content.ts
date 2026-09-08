const XML_ESCAPES: Record<string, string> = {
	"&": "&amp;",
	"<": "&lt;",
	">": "&gt;",
	'"': "&quot;",
	"'": "&apos;",
};

/** Escape untrusted text before embedding it in Supermemory XML boundaries. */
export function escapeSupermemoryXmlText(value: string): string {
	return value.replace(/[&<>"']/g, character => XML_ESCAPES[character] ?? character);
}
