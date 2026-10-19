// The local Plotly runtime is mounted separately; viewport replies contain only
// a bounded window. Keep one chart DOM node and replace its arrays on replies.
export default function render({parentElement, data, setStateValue}) {
    if (parentElement.__quantChart) {
        parentElement.__quantChart.update(data, setStateValue);
        return parentElement.__quantChart.dispose;
    }
    const chart = document.createElement("div");
    chart.classList.add("quant-bar-chart");
    chart.style.width = "100%";
    chart.style.height = "650px";
    parentElement.appendChild(chart);
    // Use the normal arrow over idle plot space; keep drag/control cursors.
    const cursorStyle = document.createElement("style");
    cursorStyle.textContent = ".quant-bar-chart .nsewdrag:not(:active) { cursor: default !important; }";
    parentElement.appendChild(cursorStyle);
    const dynamic = Boolean(data.chart_id && setStateValue);
    const chartId = data.chart_id;
    let payload = data, send = setStateValue;
    const count = dynamic ? data.total_count : data.figure.data[0].x.length;
    const fullRange = [-0.5, count - 0.5];
    const viewStorageKey = "quant-market-chart-views-v1";
    function storedViews() {
        try {
            const saved = JSON.parse(localStorage.getItem(viewStorageKey) || "[]");
            return Array.isArray(saved) ? saved : [];
        } catch (_) { return []; }
    }
    const savedView = dynamic ? storedViews().find(item => item?.chart_id === chartId) : null;
    const savedRange = Array.isArray(savedView?.range) && savedView.range.length === 2
        && savedView.range.every(Number.isFinite) && savedView.range[0] < savedView.range[1]
        ? savedView.range : null;
    let range = boundedRange(savedRange || data.view_range || data.figure.layout.xaxis.range);
    if (savedView && ["zoom", "pan"].includes(savedView.dragmode)) data.figure.layout.dragmode = savedView.dragmode;
    let wanted = range.slice();
    let latestRequested = data.request_id || 0, applied = latestRequested;
    let overlayRevision = data.overlay_revision || 0;
    let overlays = data.trend_overlays || [];
    let sampledOverlayView = "";
    let resolutionBudget = data.width_budget || Math.max(200, Math.min(1000, chart.clientWidth || 900));
    let sentBudget = resolutionBudget, observedWidth = chart.clientWidth;
    let pending = false, disposed = false, initialized = false;
    let work = Promise.resolve(), fitQueued = false, needsResize = false;
    let observer, timer, controls, status, canvas, startControl, endControl;

    function boundedRange(requested) {
        if (!requested || !requested.every(Number.isFinite)) return fullRange.slice();
        const width = Math.min(count, Math.max(1, requested[1] - requested[0]));
        const center = (requested[0] + requested[1]) / 2;
        const left = Math.max(-0.5, Math.min(count - 0.5 - width, center - width / 2));
        return [left, left + width];
    }

    function trendPrice(trend, position) {
        return Math.exp(trend.intercept + trend.slope * (position - trend.start_index));
    }

    function withOverlays(base) {
        // Overlay changes need not carry the currently displayed server window.
        // Always sample descriptors against the window we actually accepted.
        const traces = base.figure.data.filter(trace => !trace.meta?.trend_overlay);
        for (const trend of overlays) {
            const first = Math.max(trend.start_index, base.loaded_range[0]);
            const last = Math.min(trend.end_index, base.loaded_range[1]);
            if (last < first) continue;
            const size = Math.min(768, Math.max(2, Math.ceil(last - first) + 1));
            const x = Array.from({length: size}, (_, index) => first + (last - first) * index / (size - 1));
            traces.push({type: "scatter", mode: "lines", x, y: x.map(value => trendPrice(trend, value)),
                name: trend.name || trend.trend_id, line: trend.line, hovertemplate: trend.hovertemplate,
                xaxis: "x", yaxis: "y", showlegend: false,
                meta: {trend_overlay: true, trend_id: trend.trend_id,
                    lambda_values: trend.lambda_values, lambda_levels: trend.lambda_levels,
                    primary_lambda: trend.primary_lambda, origin: trend.origin}});
        }
        const layout = {...base.figure.layout,
            xaxis: {...base.figure.layout.xaxis, range: range.slice(), autorange: false},
            xaxis2: {...base.figure.layout.xaxis2, range: range.slice(), autorange: false}};
        if (initialized && chart.layout) layout.dragmode = chart.layout.dragmode;
        return {...base, overlay_revision: overlayRevision, trend_overlays: overlays,
            figure: {...base.figure, data: traces, layout}};
    }

    async function sampleVisibleOverlays() {
        const signature = [range[0], range[1], chart.clientWidth, overlayRevision, applied].join("/");
        if (signature === sampledOverlayView) return;
        sampledOverlayView = signature;
        const descriptors = new Map(overlays.map(trend => [trend.trend_id, trend]));
        const indices = [], xs = [], ys = [];
        payload.figure.data.forEach((trace, index) => {
            if (!trace.meta?.trend_overlay) return;
            const trend = descriptors.get(trace.meta.trend_id);
            if (!trend) return;
            const first = Math.max(trend.start_index, range[0]);
            const last = Math.min(trend.end_index, range[1]);
            // Plotly line hover uses nearby sampled points. Adapt the samples
            // to the visible pixels so deep zoom still permits hovering along
            // the curve. This evaluates the same fit; it never reruns analysis.
            const width = chart.clientWidth * Math.max(0, last - first) / (range[1] - range[0]);
            const size = Math.min(768, Math.max(2, Math.ceil(width / 6) + 1));
            const x = last < first ? [] : Array.from({length: size}, (_, item) => first + (last - first) * item / (size - 1));
            indices.push(index);
            xs.push(x);
            ys.push(x.map(position => trendPrice(trend, position)));
        });
        if (indices.length) await window.Plotly.restyle(chart, {x: xs, y: ys}, indices);
    }

    function drawOverview() {
        if (!dynamic || disposed) return;
        status.textContent = (pending ? "Loading visible period… · " : "") + payload.resolution
            + ` · ${count.toLocaleString()} stored bars`
            + (payload.omission_marker_note ? " · " + payload.omission_marker_note : "");
        startControl.value = String(wanted[0] + 0.5);
        endControl.value = String(wanted[1] + 0.5);
        const overview = payload.overview;
        const width = Math.max(1, chart.clientWidth), height = 55;
        const scale = window.devicePixelRatio || 1;
        canvas.width = width * scale;
        canvas.height = height * scale;
        canvas.style.height = height + "px";
        const context = canvas.getContext("2d");
        context.scale(scale, scale);
        const prices = overview.close;
        const low = Math.min(...prices), high = Math.max(...prices);
        const x = position => (position + 0.5) / count * width;
        const y = price => height - 5 - (price - low) / (high - low || 1) * (height - 10);
        context.strokeStyle = "#7894ae";
        context.lineWidth = 1.5;
        context.beginPath();
        prices.forEach((price, index) => {
            if (index) context.lineTo(x(overview.positions[index]), y(price));
            else context.moveTo(x(overview.positions[index]), y(price));
        });
        context.stroke();
        context.fillStyle = "rgba(50, 130, 210, 0.22)";
        context.fillRect(x(wanted[0]), 0, Math.max(2, x(wanted[1]) - x(wanted[0])), height);
        canvas.title = `${overview.label}: ${overview.first_label} — ${overview.last_label}`;
    }

    function makeControls() {
        controls = document.createElement("div");
        controls.style.cssText = "font:13px sans-serif;padding:0 12px;color:inherit";
        const buttons = document.createElement("div");
        buttons.style.cssText = "display:flex;gap:8px;align-items:center;flex-wrap:wrap";
        for (const [label, target] of [["All history", fullRange], ["Latest 50", [Math.max(0, count - 50) - 0.5, count - 0.5]]]) {
            const button = document.createElement("button");
            button.textContent = label;
            button.style.cssText = "cursor:pointer;padding:5px 9px;background:transparent;color:inherit;border:1px solid #7894ae;border-radius:4px";
            button.addEventListener("click", () => navigate(target));
            buttons.appendChild(button);
        }
        status = document.createElement("span");
        status.setAttribute("role", "status");
        buttons.appendChild(status);
        controls.appendChild(buttons);
        canvas = document.createElement("canvas");
        canvas.style.width = "100%";
        canvas.setAttribute("aria-label", "Full history overview; use period sliders below to navigate");
        controls.appendChild(canvas);
        const sliders = document.createElement("div");
        sliders.style.cssText = "display:flex;gap:12px";
        [startControl, endControl] = ["Start of visible period", "End of visible period"].map(label => {
            const wrapper = document.createElement("label");
            wrapper.style.cssText = "display:flex;align-items:center;gap:6px;flex:1;min-width:0";
            wrapper.append(label.startsWith("Start") ? "From" : "To");
            const input = document.createElement("input");
            input.type = "range";
            input.min = "0";
            input.max = String(count);
            input.step = "1";
            input.style.cssText = "width:100%;min-width:0";
            input.setAttribute("aria-label", label);
            wrapper.appendChild(input);
            sliders.appendChild(wrapper);
            return input;
        });
        startControl.addEventListener("input", () => navigate([
            Math.min(Number(startControl.value), wanted[1] - 0.5) - 0.5, wanted[1]]));
        endControl.addEventListener("input", () => navigate([
            wanted[0], Math.max(Number(endControl.value) - 0.5, wanted[0] + 1)]));
        controls.appendChild(sliders);
        parentElement.appendChild(controls);
    }

    function needsWindow(target) {
        if (!dynamic) return false;
        if (target[0] < payload.loaded_range[0] || target[1] > payload.loaded_range[1]) return true;
        const span = target[1] - target[0];
        const originalVisible = Math.min(count, Math.ceil(target[1] + 0.5) - Math.floor(target[0] + 0.5));
        const budget = Math.max(200, Math.min(1000, chart.clientWidth || 900));
        const grouped = payload.source_counts.some(value => value > 1);
        if (!grouped) return originalVisible > budget;
        const previousSpan = payload.view_range[1] - payload.view_range[0];
        return originalVisible <= budget || span < previousSpan * 0.65 || span > previousSpan * 1.6
            || budget < resolutionBudget * 0.7 || budget > resolutionBudget * 1.5;
    }

    function rememberView() {
        if (!dynamic) return;
        try {
            const views = storedViews().filter(item => item?.chart_id !== chartId);
            views.unshift({chart_id: chartId, range: wanted.slice(),
                dragmode: chart.layout?.dragmode || payload.figure.layout.dragmode});
            localStorage.setItem(viewStorageKey, JSON.stringify(views.slice(0, 32)));
        } catch (_) { /* Storage restrictions must not disable chart navigation. */ }
    }

    function navigate(target) {
        wanted = boundedRange(target);
        rememberView();
        clearTimeout(timer);
        // Invalidate in-flight responses immediately, including during debounce.
        if (needsWindow(wanted)) {
            latestRequested++;
            pending = true;
            timer = setTimeout(() => {
                if (disposed) return;
                sentBudget = Math.max(200, Math.min(1000, chart.clientWidth));
                send("viewport", {chart_id: chartId, request_id: latestRequested,
                    start: wanted[0], end: wanted[1], width: Math.min(10000, Math.max(1, chart.clientWidth))});
            }, 150);
            // Keep the old, correctly scaled window until new bars arrive.
        } else {
            if (pending) latestRequested++;
            pending = false;
            range = wanted.slice();
        }
        queueFit();
        drawOverview();
    }

    async function fit() {
        if (disposed || !initialized) return;
        await sampleVisibleOverlays();
        const figure = payload.figure, candles = figure.data[0], volumes = figure.data[1].y;
        const labels = figure.layout.meta.bar_labels;
        const visible = [];
        let low = Infinity, high = -Infinity, volume = 0;
        for (let index = 0; index < candles.x.length; index++) {
            const first = dynamic ? payload.first_indices[index] : index;
            const last = dynamic ? payload.last_indices[index] : index;
            if (last + 0.4 < range[0] || first - 0.4 > range[1]) continue;
            visible.push(index);
            low = Math.min(low, candles.low[index]);
            high = Math.max(high, candles.high[index]);
            volume = Math.max(volume, volumes[index]);
        }
        for (const trend of overlays) {
            const first = Math.max(trend.start_index, range[0]);
            const last = Math.min(trend.end_index, range[1]);
            if (last < first) continue;
            // A log-linear fit is monotone. Evaluate exact clipped endpoints,
            // including fractional viewport edges between rendered samples.
            for (const position of [first, last]) {
                const value = trendPrice(trend, position);
                if (Number.isFinite(value)) {
                    low = Math.min(low, value);
                    high = Math.max(high, value);
                }
            }
        }
        const tickCount = Math.max(2, Math.min(12, Math.floor(chart.clientWidth / 140)));
        const step = Math.max(1, Math.ceil(visible.length / tickCount));
        const ticks = [], text = [];
        for (let index = 0; index < visible.length; index += step) {
            const item = visible[index];
            ticks.push(candles.x[item]);
            text.push(labels[item].replace("T", "<br>"));
        }
        const update = {
            "xaxis.range": range.slice(), "xaxis.autorange": false,
            "xaxis2.range": range.slice(), "xaxis2.autorange": false,
            "xaxis.tickvals": ticks, "xaxis.ticktext": text,
            "xaxis2.tickvals": ticks, "xaxis2.ticktext": text,
        };
        if (visible.length) {
            const padding = high > low ? (high - low) * 0.05 : (Math.abs(low) * 0.01 || 1);
            Object.assign(update, {"yaxis.range": [low - padding, high + padding], "yaxis.autorange": false,
                "yaxis2.range": [0, volume > 0 ? volume * 1.08 : 1], "yaxis2.autorange": false});
        }
        await window.Plotly.relayout(chart, update);
        drawOverview();
    }

    function queueFit(resize = false) {
        if (disposed) return;
        needsResize ||= resize;
        if (fitQueued) return;
        fitQueued = true;
        work = work.then(async () => {
            fitQueued = false;
            if (disposed || !initialized) return;
            if (needsResize) {
                needsResize = false;
                await window.Plotly.Plots.resize(chart);
            }
            await fit();
        }).catch(showError);
    }

    function showError(error) {
        if (!disposed) {
            if (status) status.textContent = "Unable to draw the chart. Reload the dataset to try again.";
            else chart.textContent = "Unable to draw the chart. Reload the dataset to try again.";
            console.error(error);
        }
    }

    function onRelayout(event) {
        // Our own fitting emits a relayout too; it is not a user gesture.
        if ("xaxis.tickvals" in event) return;
        if ("dragmode" in event) rememberView();
        for (const axis of ["xaxis", "xaxis2"]) {
            if (event[axis + ".autorange"]) {
                navigate(fullRange);
                return;
            }
            const requested = event[axis + ".range"] || (
                axis + ".range[0]" in event ? [event[axis + ".range[0]"], event[axis + ".range[1]"]] : null);
            if (requested) {
                navigate(requested);
                return;
            }
        }
    }

    function update(next, callback) {
        if (disposed || next.chart_id !== chartId) return;
        send = callback || send;
        if (!dynamic) return;
        work = work.then(async () => {
            if (disposed) return;
            const newWindow = next.request_id >= latestRequested && next.request_id > applied;
            const newOverlays = (next.overlay_revision || 0) > overlayRevision;
            if (!newWindow && !newOverlays) return;
            if (newOverlays) {
                overlayRevision = next.overlay_revision;
                overlays = next.trend_overlays || [];
            }
            if (newWindow) {
                payload = next;
                applied = next.request_id;
                latestRequested = applied;
                resolutionBudget = next.width_budget || sentBudget;
                pending = false;
                range = boundedRange(next.view_range);
                wanted = range.slice();
            }
            // A late navigation response may contain older overlay settings;
            // a new overlay may contain an old navigation response. Merge these
            // independent revisions without rolling either dimension backward.
            payload = withOverlays(payload);
            sampledOverlayView = "";
            await window.Plotly.react(chart, payload.figure.data, payload.figure.layout,
                {...payload.config, responsive: false, doubleClick: "reset"});
            await fit();
            // The layout may have changed while this request was in flight.
            if (!disposed && newWindow && needsWindow(wanted)) navigate(wanted);
        }).catch(showError);
    }

    function dispose() {
        disposed = true;
        clearTimeout(timer);
        window.removeEventListener("quant-plotly-ready", start);
        observer?.disconnect();
        chart.removeAllListeners?.("plotly_relayout");
        chart.remove();
        cursorStyle.remove();
        controls?.remove();
        delete parentElement.__quantChart;
        // Finish a queued resize/react before purge removes Plotly's state.
        void work.finally(() => window.Plotly?.purge(chart));
    }

    function start() {
        if (disposed || initialized || !window.Plotly) return;
        window.removeEventListener("quant-plotly-ready", start);
        initialized = true;
        work = window.Plotly.newPlot(chart, payload.figure.data, payload.figure.layout,
            {...payload.config, responsive: false, doubleClick: "reset"}).then(() => {
            if (disposed) return;
            chart.on("plotly_relayout", onRelayout);
            if (savedRange) navigate(wanted);
            queueFit();
            observer = new ResizeObserver(() => {
                if (disposed || chart.clientWidth === 0) return;
                queueFit(true);
                if (observedWidth !== chart.clientWidth) {
                    observedWidth = chart.clientWidth;
                    if (!pending && needsWindow(wanted)) navigate(wanted);
                }
            });
            observer.observe(chart);
        }).catch(showError);
    }

    parentElement.__quantChart = {update, dispose};
    if (dynamic) {
        payload = withOverlays(payload);
        makeControls();
    }
    if (window.Plotly) start();
    else window.addEventListener("quant-plotly-ready", start);
    return dispose;
}
