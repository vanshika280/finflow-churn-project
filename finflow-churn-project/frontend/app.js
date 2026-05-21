const paths = {
  scores: "../outputs/churn_probability_scores.csv",
  segments: "../outputs/prioritized_customer_segments.csv",
  top20: "../outputs/top_20_save_list.csv",
  featureImportance: "../outputs/feature_importance.csv",
  metadata: "../models/model_metadata.json",
};

const colors = {
  high: "#fb7185",
  medium: "#fbbf24",
  low: "#2dd4bf",
  blue: "#60a5fa",
  teal: "#2dd4bf",
  gold: "#fbbf24",
  ink: "#f8fafc",
  muted: "#d1dbe8",
  line: "#3a4a63",
  surface: "#111827",
  track: "#2a3a52",
};

const currency = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

const numberFormat = new Intl.NumberFormat("en-IN");
const percentFormat = new Intl.NumberFormat("en-IN", {
  style: "percent",
  maximumFractionDigits: 1,
});

let dashboardData = null;

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      value += '"';
      i += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") {
        i += 1;
      }
      row.push(value);
      if (row.some((cell) => cell.trim() !== "")) {
        rows.push(row);
      }
      row = [];
      value = "";
    } else {
      value += char;
    }
  }

  if (value || row.length) {
    row.push(value);
    rows.push(row);
  }

  const headers = rows.shift().map((header) => header.trim());
  return rows.map((cells) => {
    const record = {};
    headers.forEach((header, index) => {
      record[header] = (cells[index] ?? "").trim();
    });
    return record;
  });
}

async function fetchCsv(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  return parseCsv(await response.text());
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  return response.json();
}

function toNumber(value) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function countBy(rows, key) {
  return rows.reduce((acc, row) => {
    const value = row[key] || "Unknown";
    acc[value] = (acc[value] || 0) + 1;
    return acc;
  }, {});
}

function sum(rows, key) {
  return rows.reduce((total, row) => total + toNumber(row[key]), 0);
}

function canvasContext(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  if (!canvas.dataset.baseHeight) {
    canvas.dataset.baseHeight = canvas.getAttribute("height") || "300";
  }
  const cssHeight = Number.parseInt(canvas.dataset.baseHeight, 10) || 300;
  canvas.width = Math.max(320, Math.floor(rect.width * ratio));
  canvas.height = Math.max(240, Math.floor(cssHeight * ratio));
  canvas.style.height = `${cssHeight}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {
    ctx,
    width: canvas.width / ratio,
    height: canvas.height / ratio,
  };
}

function clearCanvas(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
}

function drawText(ctx, text, x, y, options = {}) {
  if (options.shadow) {
    ctx.shadowColor = "rgba(0, 0, 0, 0.55)";
    ctx.shadowBlur = 4;
    ctx.shadowOffsetY = 1;
  } else {
    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;
  }
  ctx.fillStyle = options.color || colors.ink;
  ctx.font = `${options.weight || 700} ${options.size || 12}px Inter, system-ui, sans-serif`;
  ctx.textAlign = options.align || "left";
  ctx.textBaseline = options.baseline || "alphabetic";
  ctx.fillText(text, x, y);
  ctx.shadowColor = "transparent";
}

function drawDonut(canvas, items) {
  const { ctx, width, height } = canvasContext(canvas);
  clearCanvas(ctx, width, height);

  const total = items.reduce((acc, item) => acc + item.value, 0);
  const radius = Math.min(width, height) * 0.3;
  const innerRadius = radius * 0.58;
  const cx = width * 0.34;
  const cy = height * 0.5;
  let angle = -Math.PI / 2;

  items.forEach((item) => {
    const slice = total ? (item.value / total) * Math.PI * 2 : 0;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = item.color;
    ctx.fill();
    angle += slice;
  });

  ctx.beginPath();
  ctx.arc(cx, cy, innerRadius, 0, Math.PI * 2);
  ctx.fillStyle = colors.surface;
  ctx.fill();

  drawText(ctx, numberFormat.format(total), cx, cy - 4, {
    align: "center",
    size: 28,
    weight: 900,
    shadow: true,
  });
  drawText(ctx, "customers", cx, cy + 18, {
    align: "center",
    size: 12,
    color: colors.muted,
  });

  const legendX = width * 0.66;
  let legendY = height * 0.34;
  items.forEach((item) => {
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, legendY - 10, 12, 12);
    drawText(ctx, item.label, legendX + 20, legendY, { size: 13 });
    drawText(
      ctx,
      `${numberFormat.format(item.value)} (${percentFormat.format(item.value / total)})`,
      legendX + 20,
      legendY + 18,
      { size: 12, color: colors.muted, weight: 700 },
    );
    legendY += 52;
  });
}

function drawVerticalBar(canvas, items) {
  const { ctx, width, height } = canvasContext(canvas);
  clearCanvas(ctx, width, height);

  const pad = { top: 20, right: 18, bottom: 54, left: 48 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const maxValue = Math.max(...items.map((item) => item.value), 1);
  const barW = chartW / items.length / 1.7;

  ctx.strokeStyle = colors.line;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + chartH - (chartH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    drawText(ctx, Math.round((maxValue * i) / 4), pad.left - 10, y + 4, {
      align: "right",
      size: 11,
      color: colors.muted,
      weight: 600,
    });
  }

  items.forEach((item, index) => {
    const slot = chartW / items.length;
    const x = pad.left + slot * index + slot / 2 - barW / 2;
    const barH = (item.value / maxValue) * chartH;
    const y = pad.top + chartH - barH;
    ctx.fillStyle = item.color;
    ctx.fillRect(x, y, barW, barH);
    drawText(ctx, numberFormat.format(item.value), x + barW / 2, y - 8, {
      align: "center",
      size: 12,
      weight: 800,
    });
    drawText(ctx, item.label, x + barW / 2, height - 22, {
      align: "center",
      size: 12,
      color: colors.muted,
      weight: 700,
    });
  });
}

function drawHorizontalBar(canvas, items, valueFormatter = numberFormat.format) {
  const { ctx, width, height } = canvasContext(canvas);
  clearCanvas(ctx, width, height);

  const pad = { top: 10, right: 22, bottom: 16, left: 178 };
  const chartW = width - pad.left - pad.right;
  const rowH = (height - pad.top - pad.bottom) / items.length;
  const maxValue = Math.max(...items.map((item) => item.value), 1);

  items.forEach((item, index) => {
    const y = pad.top + rowH * index + rowH * 0.22;
    const barH = Math.max(12, rowH * 0.44);
    const barW = (item.value / maxValue) * chartW;

    drawText(ctx, item.label, pad.left - 12, y + barH / 2 + 4, {
      align: "right",
      size: 12,
      color: colors.ink,
      weight: 700,
    });

    ctx.fillStyle = colors.track;
    ctx.fillRect(pad.left, y, chartW, barH);
    ctx.fillStyle = item.color || colors.teal;
    ctx.fillRect(pad.left, y, barW, barH);

    drawText(ctx, valueFormatter(item.value), pad.left + barW + 8, y + barH / 2 + 4, {
      size: 12,
      color: colors.muted,
      weight: 800,
    });
  });
}

function drawHistogram(canvas, values) {
  const bins = Array.from({ length: 10 }, (_, index) => ({
    label: `${index * 10}-${index * 10 + 10}%`,
    value: 0,
    color: index >= 7 ? colors.high : index >= 4 ? colors.medium : colors.low,
  }));

  values.forEach((value) => {
    const index = Math.min(9, Math.floor(value * 10));
    bins[index].value += 1;
  });

  drawVerticalBar(canvas, bins);
}

function drawScatter(canvas, rows) {
  const { ctx, width, height } = canvasContext(canvas);
  clearCanvas(ctx, width, height);

  const pad = { top: 18, right: 18, bottom: 42, left: 64 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const maxClv = Math.max(...rows.map((row) => toNumber(row.CLV)), 1);

  ctx.strokeStyle = colors.line;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + chartH - (chartH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    drawText(ctx, currency.format((maxClv * i) / 4), pad.left - 10, y + 4, {
      align: "right",
      size: 10,
      color: colors.muted,
      weight: 600,
    });
  }

  ctx.strokeStyle = "#cbd5e1";
  ctx.beginPath();
  ctx.moveTo(pad.left + chartW * 0.65, pad.top);
  ctx.lineTo(pad.left + chartW * 0.65, pad.top + chartH);
  ctx.stroke();

  rows.forEach((row) => {
    const probability = toNumber(row.Churn_Probability);
    const clv = toNumber(row.CLV);
    const x = pad.left + probability * chartW;
    const y = pad.top + chartH - (clv / maxClv) * chartH;
    ctx.beginPath();
    ctx.arc(x, y, row.Priority_Segment === "High Priority" ? 4.2 : 2.8, 0, Math.PI * 2);
    ctx.fillStyle =
      row.Priority_Segment === "High Priority"
        ? colors.high
        : row.Priority_Segment === "Medium Priority"
          ? colors.medium
          : colors.low;
    ctx.globalAlpha = row.Priority_Segment === "High Priority" ? 0.8 : 0.42;
    ctx.fill();
  });
  ctx.globalAlpha = 1;

  drawText(ctx, "Churn probability", pad.left + chartW / 2, height - 10, {
    align: "center",
    size: 12,
    color: colors.muted,
    weight: 700,
  });
  drawText(ctx, "0%", pad.left, height - 22, { size: 11, color: colors.muted });
  drawText(ctx, "100%", width - pad.right, height - 22, {
    align: "right",
    size: 11,
    color: colors.muted,
  });
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function riskItems(scores) {
  const counts = countBy(scores, "Risk_Level");
  return [
    { label: "High Risk", value: counts["High Risk"] || 0, color: colors.high },
    { label: "Medium Risk", value: counts["Medium Risk"] || 0, color: colors.medium },
    { label: "Low Risk", value: counts["Low Risk"] || 0, color: colors.low },
  ];
}

function priorityItems(segments) {
  const counts = countBy(segments, "Priority_Segment");
  return [
    { label: "High Priority", value: counts["High Priority"] || 0, color: colors.high },
    { label: "Medium Priority", value: counts["Medium Priority"] || 0, color: colors.medium },
    { label: "Low Priority", value: counts["Low Priority"] || 0, color: colors.low },
  ];
}

function renderKpis(data) {
  const highRisk = data.scores.filter((row) => row.Risk_Level === "High Risk").length;
  const highPriority = data.segments.filter(
    (row) => row.Priority_Segment === "High Priority",
  ).length;
  const top20Clv = sum(data.top20, "CLV");

  setText("totalCustomers", numberFormat.format(data.scores.length));
  setText("highRiskCustomers", numberFormat.format(highRisk));
  setText("highPriorityCustomers", numberFormat.format(highPriority));
  setText("top20Clv", currency.format(top20Clv));

  const bestModel = data.metadata.best_model || "Selected model";
  const metrics = data.metadata.metrics?.[bestModel] || {};
  document.getElementById("modelPill").textContent =
    `${bestModel} selected | F1 ${toNumber(metrics.f1_score).toFixed(3)} | ROC-AUC ${toNumber(metrics.roc_auc).toFixed(3)}`;
}

function retentionReason(row) {
  if (row.Primary_Retention_Reason) {
    return row.Primary_Retention_Reason;
  }
  if (row.Risk_Level === "High Risk") {
    return "High churn probability";
  }
  if (row.Risk_Level === "Medium Risk") {
    return "Moderate churn probability";
  }
  return "Lower immediate save priority";
}

function tableRowsForSearch(data) {
  const search = document.getElementById("customerSearch").value.trim().toLowerCase();
  if (!search) {
    document.getElementById("tableStatus").textContent =
      "Top 20 customers for immediate campaign outreach.";
    return data.top20;
  }

  const top20ByCustomer = new Map(
    data.top20.map((row) => [row.Customer_ID, row]),
  );
  const filtered = data.segments
    .map((row, index) => ({
      ...row,
      Rank: index + 1,
      Primary_Retention_Reason:
        top20ByCustomer.get(row.Customer_ID)?.Primary_Retention_Reason ||
        retentionReason(row),
    }))
    .filter((row) => row.Customer_ID.toLowerCase().includes(search));

  document.getElementById("tableStatus").textContent =
    filtered.length > 0
      ? `Showing ${filtered.length} matching customer${filtered.length === 1 ? "" : "s"} from all scored customers.`
      : "No matching customer found in the scored dataset.";

  return filtered;
}

function renderCustomerTable(data) {
  const rows = tableRowsForSearch(data);
  const tbody = document.getElementById("top20Table");
  if (rows.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" class="empty-state">No matching customer found. Try an ID like CUST10652.</td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = rows
    .map((row) => {
      const segmentClass =
        row.Priority_Segment === "High Priority"
          ? "segment-high"
          : row.Priority_Segment === "Medium Priority"
            ? "segment-medium"
            : "segment-low";

      return `
        <tr>
          <td>${row.Rank}</td>
          <td><strong>${row.Customer_ID}</strong></td>
          <td>${percentFormat.format(toNumber(row.Churn_Probability))}</td>
          <td>${currency.format(toNumber(row.CLV))}</td>
          <td>${numberFormat.format(Math.round(toNumber(row.Priority_Score)))}</td>
          <td><span class="segment-badge ${segmentClass}">${row.Priority_Segment}</span></td>
          <td>${retentionReason(row)}</td>
        </tr>
      `;
    })
    .join("");
}

function renderCharts(data) {
  drawDonut(document.getElementById("riskDonut"), riskItems(data.scores));
  drawVerticalBar(document.getElementById("priorityBar"), priorityItems(data.segments));
  drawHistogram(
    document.getElementById("probabilityHistogram"),
    data.scores.map((row) => toNumber(row.Churn_Probability)),
  );
  drawScatter(document.getElementById("clvScatter"), data.segments);

  const modelName = data.metadata.best_model || "Gradient Boosting";
  const importance = data.featureImportance
    .filter((row) => row.Model === modelName)
    .sort((a, b) => toNumber(b.Importance) - toNumber(a.Importance))
    .slice(0, 7)
    .map((row) => ({
      label: row.Feature.replaceAll("_", " "),
      value: toNumber(row.Importance),
      color: colors.teal,
    }));
  drawHorizontalBar(
    document.getElementById("featureImportance"),
    importance,
    (value) => value.toFixed(3),
  );

  const topCustomers = data.top20
    .slice(0, 10)
    .reverse()
    .map((row) => ({
      label: row.Customer_ID,
      value: toNumber(row.Priority_Score),
      color: colors.blue,
    }));
  drawHorizontalBar(
    document.getElementById("topCustomers"),
    topCustomers,
    (value) => numberFormat.format(Math.round(value)),
  );

  const reasonItems = Object.entries(countBy(data.top20, "Primary_Retention_Reason")).map(
    ([label, value]) => ({ label, value, color: colors.gold }),
  );
  drawHorizontalBar(document.getElementById("reasonBar"), reasonItems);
}

function showToast(message, isError = false) {
  const toast = document.getElementById("statusToast");
  toast.textContent = message;
  toast.style.borderLeftColor = isError ? colors.high : colors.teal;
  toast.classList.remove("is-hidden");

  if (!isError) {
    window.setTimeout(() => toast.classList.add("is-hidden"), 2400);
  }
}

async function initDashboard() {
  try {
    const [scores, segments, top20, featureImportance, metadata] = await Promise.all([
      fetchCsv(paths.scores),
      fetchCsv(paths.segments),
      fetchCsv(paths.top20),
      fetchCsv(paths.featureImportance),
      fetchJson(paths.metadata),
    ]);

    dashboardData = { scores, segments, top20, featureImportance, metadata };
    renderKpis(dashboardData);
    renderCharts(dashboardData);
    renderCustomerTable(dashboardData);
    showToast("Dashboard loaded from model outputs.");
  } catch (error) {
    showToast(`${error.message}. Start a local server from the project root.`, true);
    console.error(error);
  }
}

document.getElementById("customerSearch").addEventListener("input", () => {
  if (dashboardData) {
    renderCustomerTable(dashboardData);
  }
});

let resizeTimer = null;
window.addEventListener("resize", () => {
  if (!dashboardData) {
    return;
  }
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => renderCharts(dashboardData), 120);
});

initDashboard();
