"""Browser half of the end-to-end test; run_e2e.py runs it inside the Playwright image.

    python ui_flow.py BASE_URL IMAGE_PATH workflow            -> prints JSON observations
    python ui_flow.py BASE_URL IMAGE_PATH after_restart SAVED_RESULT_ID
    python ui_flow.py BASE_URL IMAGE_PATH clear

The UI shows no cache information, so cache outcomes are read from the API by result id.
"""
import asyncio
import json
import re
import sys

from playwright.async_api import async_playwright, expect

BASE, IMAGE, PHASE = sys.argv[1], sys.argv[2], sys.argv[3]


async def current_id(page):
    return await page.get_by_test_id("current-result").get_attribute("data-result-id")


async def segment(page, expect_new_after=None, **fields):
    """Fills fields, presses Segment and waits for a different result to be displayed."""
    for name, value in fields.items():
        await page.locator(f"input[name={name}]").fill(str(value))
    await page.locator("form button[type=submit]").click()
    if expect_new_after is None:
        await expect(page.get_by_test_id("current-result")).to_be_visible(timeout=60_000)
    else:
        await expect(page.get_by_test_id("current-result")).not_to_have_attribute(
            "data-result-id", expect_new_after, timeout=60_000)
    await expect(page.locator("form button[type=submit]")).to_be_enabled(timeout=60_000)
    current = page.get_by_test_id("current-result")
    rid = await current_id(page)
    stored = await (await page.request.get(f"{BASE}/api/results/{rid}")).json()
    return {
        "id": rid,
        "cache": stored["data"]["metrics"]["cache"]["graph"],
        "segments": int(await current.get_by_test_id("segments").inner_text()),
        "stored_segments": stored["data"]["metrics"]["segments"],
        "k": stored["data"]["params"]["k"],
        "method": stored["data"]["params"]["method"],
    }


async def ui_facts(page):
    text = await page.locator("body").inner_text()
    return {
        "mentions_cache": bool(re.search(r"\b(cache[d]?|hit|miss|reused)\b", text, re.IGNORECASE)),
        "removed_inputs": await page.locator("input[name=seed], input[name=sigma_color], "
                                             "input[name=spatial_scale], select[name=method]").count(),
        "sliders": sorted([await r.get_attribute("id") for r in await page.locator("input[type=range]").all()]),
        "ids_shown": [await e.inner_text() for e in await page.get_by_test_id("result-id").all()],
        "run_info": await page.get_by_test_id("run-info").inner_text(),
    }


async def image_loaded(locator):
    await expect(locator).to_be_visible()
    await locator.evaluate("img => img.decode()")
    return await locator.evaluate("img => img.naturalWidth > 0")


async def workflow(page):
    out = {}
    await page.goto(BASE)
    await expect(page.get_by_role("heading", name="Spectral image segmentation")).to_be_visible()
    await page.get_by_label("Image file").set_input_files(IMAGE)
    await expect(page.get_by_test_id("run-info")).to_contain_text("downtown_toronto.jpeg")

    out["first"] = await segment(page, superpixels=400, k=6)
    out["k_changed"] = await segment(page, expect_new_after=out["first"]["id"], k=8)
    out["graph_changed"] = await segment(page, expect_new_after=out["k_changed"]["id"], superpixels=500)
    out["ui"] = await ui_facts(page)
    out["overlay_loaded"] = await image_loaded(page.get_by_test_id("current-result").get_by_test_id("overlay-image"))

    # Stale response: the first of two quick requests is held back until after the second
    # has been displayed. It must not replace the newer result.
    held = asyncio.Event()
    released = []

    async def hold_first(route):
        if not held.is_set():
            held.set()
            await asyncio.sleep(2.0)
            released.append(True)
            try:
                await route.continue_()
            except Exception:
                pass  # the browser already aborted this request
        else:
            await route.continue_()

    await page.route("**/api/runs/*/results", hold_first)
    responses = []
    page.on("response", lambda r: responses.append(r) if r.url.endswith("/results") else None)
    before = await current_id(page)
    await page.locator("input[name=k]").fill("4")
    await page.locator("form button[type=submit]").click()
    await held.wait()
    await page.locator("input[name=k]").fill("5")
    await page.locator("form button[type=submit]").click()
    await expect(page.get_by_test_id("current-result")).not_to_have_attribute("data-result-id", before, timeout=60_000)
    await asyncio.sleep(3.0)  # let the held request finish (or fail) after the newer one
    await page.unroute("**/api/runs/*/results")
    out["stale"] = {
        "held_released": bool(released),
        "displayed_segments": int(await page.get_by_test_id("current-result").get_by_test_id("segments").inner_text()),
        "displayed_id": await current_id(page),
        "k_field": await page.locator("input[name=k]").input_value(),
    }
    # The response to the newer request (k=5), whatever order the responses arrived in.
    newest = None
    for r in responses:
        if r.request.method == "POST" and r.ok:
            data = (await r.json())["data"]
            if data["params"]["k"] == 5:
                newest = data["result_id"]
    out["stale"]["newest_response_id"] = newest

    # Save references the displayed result, never another one.
    displayed = await current_id(page)
    async with page.expect_request(lambda r: r.method == "PUT") as put:
        await page.get_by_test_id("save").click()
    out["save"] = {"displayed": displayed, "put_url": (await put.value).url}
    await expect(page.get_by_test_id("saved-badge")).to_be_visible()
    row = page.get_by_test_id(f"history-{displayed}")
    await expect(row).to_be_visible()
    await row.click()
    viewing = page.get_by_test_id("viewing")
    await expect(viewing).to_contain_text("Saved experiment")
    out["viewing"] = {
        "image_loaded": await image_loaded(viewing.get_by_test_id("overlay-image")),
        "current_unchanged": await current_id(page) == displayed,
        "segments": int(await viewing.get_by_test_id("segments").inner_text()),
    }
    return out


async def after_restart(page, saved_id):
    out = {}
    await page.goto(BASE)
    row = page.get_by_test_id(f"history-{saved_id}")
    await expect(row).to_be_visible(timeout=30_000)
    await row.click()
    viewing = page.get_by_test_id("viewing")
    out["viewing_image_loaded"] = await image_loaded(viewing.get_by_test_id("overlay-image"))
    out["viewing_segments"] = int(await viewing.get_by_test_id("segments").inner_text())
    await page.get_by_label("Image file").set_input_files(IMAGE)
    await expect(page.get_by_test_id("run-info")).to_contain_text("downtown_toronto.jpeg")
    out["recompute"] = await segment(page, superpixels=400, k=5)
    return out


async def clear(page):
    """Clearing the log deletes saved experiments but keeps the cached graph."""
    out = {}
    page.on("dialog", lambda d: asyncio.ensure_future(d.accept()))
    await page.goto(BASE)
    await page.get_by_label("Image file").set_input_files(IMAGE)
    await expect(page.get_by_test_id("run-info")).to_contain_text("downtown_toronto.jpeg")
    first = await segment(page, superpixels=400, k=6)
    await page.get_by_test_id("save").click()
    await expect(page.get_by_test_id("saved-badge")).to_be_visible()
    await expect(page.get_by_test_id(f"history-{first['id']}")).to_be_visible()
    out["history_headers"] = [await h.inner_text() for h in await page.locator("[data-testid=history] th").all()]
    out["rows_before"] = await page.locator("[data-testid=history] tbody tr").count()
    await page.get_by_test_id("clear-history").click()
    await expect(page.get_by_text("Nothing saved yet.")).to_be_visible()
    out["notice"] = await page.locator(".notice").inner_text()
    out["current_cleared"] = await page.get_by_test_id("current-result").count() == 0  # it was saved
    out["after_clear"] = await segment(page, k=7)
    return out


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        phases = {"workflow": lambda: workflow(page), "after_restart": lambda: after_restart(page, sys.argv[4]),
                  "clear": lambda: clear(page)}
        result = await phases[PHASE]()
        result["page_errors"] = errors
        await browser.close()
    print(json.dumps(result))


asyncio.run(main())
