// Plotly is bundled by chart_component.py. No fetches or Python reruns on zoom.
export default function render({parentElement, data}) {
    const chart = document.createElement("div");
    chart.style.width = "100%";
    chart.style.height = "650px";
    parentElement.appendChild(chart);
    const {figure, config} = data;
    const labels = figure.layout.meta.bar_labels;
    const candles = figure.data[0];
    const volumes = figure.data[1].y;
    const count = labels.length;
    const fullRange = [-0.5, count - 0.5];
    let range = boundedRange(figure.layout.xaxis.range);
    let disposed = false;
    let work = Promise.resolve();
    let fitQueued = false;
    let needsResize = false;
    let observer;

    function boundedRange(requested) {
        if (!requested || !requested.every(Number.isFinite)) return fullRange.slice();
        const width = Math.min(count, Math.max(1, requested[1] - requested[0]));
        const center = (requested[0] + requested[1]) / 2;
        const left = Math.max(-0.5, Math.min(count - 0.5 - width, center - width / 2));
        return [left, left + width];
    }

    async function fit() {
        if (disposed) return;
        // Include partially visible candle bodies at the window edges.
        const first = Math.max(0, Math.ceil(range[0] - 0.4));
        const last = Math.min(count - 1, Math.floor(range[1] + 0.4));
        let low = Infinity, high = -Infinity, volume = 0;
        for (let i = first; i <= last; i++) {
            low = Math.min(low, candles.low[i]);
            high = Math.max(high, candles.high[i]);
            volume = Math.max(volume, volumes[i]);
        }
        const padding = high > low ? (high - low) * 0.05 : (Math.abs(low) * 0.01 || 1);
        const tickCount = Math.max(2, Math.min(12, Math.floor(chart.clientWidth / 140)));
        const step = Math.max(1, Math.ceil((last - first + 1) / tickCount));
        const ticks = [], text = [];
        for (let i = first; i <= last; i += step) {
            ticks.push(i);
            text.push(labels[i].replace("T", "<br>"));
        }
        await Plotly.relayout(chart, {
            "xaxis.range": range.slice(), "xaxis.autorange": false,
            "xaxis2.range": range.slice(), "xaxis2.autorange": false,
            "xaxis.tickvals": ticks, "xaxis.ticktext": text,
            "xaxis2.tickvals": ticks, "xaxis2.ticktext": text,
            "yaxis.range": [low - padding, high + padding], "yaxis.autorange": false,
            "yaxis2.range": [0, volume > 0 ? volume * 1.08 : 1], "yaxis2.autorange": false,
        });
    }

    function queueFit(resize = false) {
        if (disposed) return;
        needsResize ||= resize;
        if (fitQueued) return;
        fitQueued = true;
        // Serialize Plotly operations and coalesce gestures arriving together.
        work = work.then(async () => {
            fitQueued = false;
            if (disposed) return;
            if (needsResize) {
                needsResize = false;
                await Plotly.Plots.resize(chart);
            }
            await fit();
        }).catch(showError);
    }

    function showError(error) {
        if (!disposed) {
            chart.textContent = "Unable to draw the chart. Reload the dataset to try again.";
            console.error(error);
        }
    }

    function onRelayout(event) {
        // Our own y-axis/tick update also emits relayout; never recurse on it.
        if ("yaxis.range" in event && "xaxis.tickvals" in event) return;
        for (const axis of ["xaxis", "xaxis2"]) {
            if (event[axis + ".autorange"]) {
                range = fullRange.slice();
                queueFit();
                return;
            }
            const requested = event[axis + ".range"] || (
                axis + ".range[0]" in event ? [event[axis + ".range[0]"], event[axis + ".range[1]"]] : null);
            if (requested) {
                range = boundedRange(requested);
                queueFit();
                return;
            }
        }
    }

    // ResizeObserver handles both window and sidebar changes through our queue.
    work = Plotly.newPlot(chart, figure.data, figure.layout, {...config, responsive: false, doubleClick: "reset"}).then(() => {
        if (disposed) return;
        chart.on("plotly_relayout", onRelayout);
        queueFit();
        observer = new ResizeObserver(() => {
            if (disposed || chart.clientWidth === 0) return;
            queueFit(true);
        });
        observer.observe(chart);
    }).catch(showError);

    return () => {
        disposed = true;
        observer?.disconnect();
        chart.removeAllListeners?.("plotly_relayout");
        chart.remove();
        // A delayed resize must finish before purge deletes Plotly's state.
        void work.finally(() => Plotly.purge(chart));
    };
}
