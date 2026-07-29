export type TodoProjectionStatus = "pending" | "in_progress" | "completed" | "failed" | "cancelled" | "abandoned";

/** A host-rendered todo owned by an extension namespace. */
export interface TodoProjectionItem {
	/** Stable within the namespace for the lifetime of the projected work item. */
	readonly id: string;
	readonly content: string;
	readonly status: TodoProjectionStatus;
}

/** A display group for extension-owned projected todos. */
export interface TodoProjectionPhase {
	/** Stable within the namespace for the lifetime of the projected phase. */
	readonly id: string;
	readonly name: string;
	readonly tasks: readonly TodoProjectionItem[];
}

/** Read-only snapshot exposed by the host for rendering and inspection. */
export interface NamespacedTodoProjection {
	readonly namespace: string;
	readonly phases: readonly TodoProjectionPhase[];
}

const TODO_PROJECTION_STATUSES: Record<TodoProjectionStatus, true> = {
	pending: true,
	in_progress: true,
	completed: true,
	failed: true,
	cancelled: true,
	abandoned: true,
};

function requireStableId(value: string, kind: "phase" | "task"): string {
	const id = value.trim();
	if (!id) throw new Error(`Todo projection ${kind} id must be non-empty`);
	return id;
}

export function normalizeTodoProjectionNamespace(namespace: string): string {
	const normalized = namespace.trim();
	if (!normalized) throw new Error("Todo projection namespace must be non-empty");
	return normalized;
}

function assertDenseTodoProjectionArray(values: readonly unknown[], label: "phase" | "task"): void {
	for (let index = 0; index < values.length; index++) {
		if (!Object.hasOwn(values, index)) {
			throw new Error(`Todo projection ${label} array must not contain holes (missing index ${index})`);
		}
	}
}

/**
 * Validate and clone extension-owned data at the public API boundary. The
 * returned value contains only fields the host renders; extra caller metadata
 * is deliberately discarded.
 */
export function cloneTodoProjection(phases: readonly TodoProjectionPhase[]): TodoProjectionPhase[] {
	const phaseIds = new Set<string>();
	const taskIds = new Set<string>();
	assertDenseTodoProjectionArray(phases, "phase");
	for (const phase of phases) assertDenseTodoProjectionArray(phase.tasks, "task");
	return phases.map(phase => {
		const id = requireStableId(phase.id, "phase");
		if (phaseIds.has(id)) throw new Error(`Duplicate todo projection phase id: ${id}`);
		phaseIds.add(id);
		const name = phase.name.trim();
		if (!name) throw new Error(`Todo projection phase ${id} must have a non-empty name`);
		const tasks = phase.tasks.map(task => {
			const taskId = requireStableId(task.id, "task");
			if (taskIds.has(taskId)) throw new Error(`Duplicate todo projection task id: ${taskId}`);
			taskIds.add(taskId);
			const content = task.content.trim();
			if (!content) throw new Error(`Todo projection task ${taskId} must have non-empty content`);
			if (TODO_PROJECTION_STATUSES[task.status] !== true) {
				throw new Error(`Invalid todo projection status for ${taskId}: ${String(task.status)}`);
			}
			return { id: taskId, content, status: task.status };
		});
		return { id, name, tasks };
	});
}

export function cloneNamespacedTodoProjections(
	projections: ReadonlyMap<string, readonly TodoProjectionPhase[]>,
): NamespacedTodoProjection[] {
	return [...projections.entries()]
		.sort(([left], [right]) => left.localeCompare(right))
		.map(([namespace, phases]) => ({ namespace, phases: cloneTodoProjection(phases) }));
}

function todoProjectionPhasesEqual(
	left: readonly TodoProjectionPhase[],
	right: readonly TodoProjectionPhase[],
): boolean {
	if (left.length !== right.length) return false;
	for (let phaseIndex = 0; phaseIndex < left.length; phaseIndex++) {
		const leftPhase = left[phaseIndex]!;
		const rightPhase = right[phaseIndex]!;
		if (leftPhase.id !== rightPhase.id || leftPhase.name !== rightPhase.name) return false;
		if (leftPhase.tasks.length !== rightPhase.tasks.length) return false;
		for (let taskIndex = 0; taskIndex < leftPhase.tasks.length; taskIndex++) {
			const leftTask = leftPhase.tasks[taskIndex]!;
			const rightTask = rightPhase.tasks[taskIndex]!;
			if (
				leftTask.id !== rightTask.id ||
				leftTask.content !== rightTask.content ||
				leftTask.status !== rightTask.status
			) {
				return false;
			}
		}
	}
	return true;
}

/**
 * Session-owned store for display-only projections. It has no dependency on
 * canonical todo state or transcript persistence.
 */
export class TodoProjectionStore {
	#projections = new Map<string, TodoProjectionPhase[]>();

	set(namespace: string, phases: readonly TodoProjectionPhase[] | undefined): boolean {
		const key = normalizeTodoProjectionNamespace(namespace);
		if (phases === undefined) return this.#projections.delete(key);
		const next = cloneTodoProjection(phases);
		const current = this.#projections.get(key);
		if (current && todoProjectionPhasesEqual(current, next)) return false;
		this.#projections.set(key, next);
		return true;
	}

	snapshot(): NamespacedTodoProjection[] {
		return cloneNamespacedTodoProjections(this.#projections);
	}

	clear(): boolean {
		if (this.#projections.size === 0) return false;
		this.#projections.clear();
		return true;
	}
}
