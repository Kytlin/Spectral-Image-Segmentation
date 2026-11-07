// Runs async tasks so that only the most recently started one may deliver a result.
// Starting a task aborts the previous one; a superseded task resolves to { current: false }
// even if its response arrives later, so a slow old response can never overwrite newer state.
export function createLatest() {
  let generation = 0;
  let controller = null;

  return {
    async run(task) {
      const mine = ++generation;
      controller?.abort();
      controller = new AbortController();
      try {
        const value = await task(controller.signal);
        return mine === generation ? { current: true, value } : { current: false };
      } catch (error) {
        if (mine !== generation || error?.name === "AbortError") return { current: false };
        throw error;
      }
    },
    // Supersedes whatever is running (e.g. a new image was uploaded).
    invalidate() {
      ++generation;
      controller?.abort();
      controller = null;
    },
  };
}
