/**
 * script.js  –  MemeGuard AI  •  Yellow Ochre Theme
 * ─────────────────────────────────────────────────────────────
 * Features:
 *   • Drag-and-drop image upload
 *   • Click-to-browse file picker
 *   • Ctrl+V / Cmd+V  paste image from clipboard  ← NEW
 *   • Image preview with remove button + source label
 *   • Character counter on textarea
 *   • POST /predict API call with FormData
 *   • Animated loading steps
 *   • Result rendering (verdict, confidence bars, prob bars)
 *   • Full error handling + retry
 *   • Reset / clear
 */

"use strict";

/* ── DOM refs ────────────────────────────────────────────────── */
const dropZone       = document.getElementById("dropZone");
const fileInput      = document.getElementById("fileInput");
const imgPreviewWrap = document.getElementById("imgPreviewWrap");
const imgPreview     = document.getElementById("imgPreview");
const removeImgBtn   = document.getElementById("removeImgBtn");
const imgBadge       = document.getElementById("imgBadge");
const imgSourceTag   = document.getElementById("imgSourceTag");

const memeText  = document.getElementById("memeText");
const charCount = document.getElementById("charCount");

const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn   = document.getElementById("resetBtn");

const resultIdle    = document.getElementById("resultIdle");
const resultLoading = document.getElementById("resultLoading");
const resultOutput  = document.getElementById("resultOutput");
const resultError   = document.getElementById("resultError");
const errorMsg      = document.getElementById("errorMsg");
const errorRetryBtn = document.getElementById("errorRetryBtn");

const verdictBanner = document.getElementById("verdictBanner");
const verdictIcon   = document.getElementById("verdictIcon");
const verdictLabel  = document.getElementById("verdictLabel");
const verdictSub    = document.getElementById("verdictSub");

const confPct     = document.getElementById("confPct");
const confBarFill = document.getElementById("confBarFill");

const hatefulFill    = document.getElementById("hatefulFill");
const notHatefulFill = document.getElementById("notHatefulFill");
const hatefulPct     = document.getElementById("hatefulPct");
const notHatefulPct  = document.getElementById("notHatefulPct");

const resultThumbWrap = document.getElementById("resultThumbWrap");
const resultThumb     = document.getElementById("resultThumb");

const lsItems = [
  document.getElementById("ls1"),
  document.getElementById("ls2"),
  document.getElementById("ls3"),
  document.getElementById("ls4"),
];

/* ── State ───────────────────────────────────────────────────── */
let selectedFile = null;
let loadingTimer = null;

/* ═══════════════════════════════════════════════════════════════
   FILE HELPERS
   ═══════════════════════════════════════════════════════════════ */

const ALLOWED_TYPES = [
  "image/png","image/jpeg","image/jpg",
  "image/gif","image/webp","image/bmp",
];

/**
 * Accept a File, validate it, preview it, and store it.
 * @param {File}   file
 * @param {string} source  "upload" | "paste" | "drop"
 */
function setFile(file, source = "upload") {
  if (!ALLOWED_TYPES.includes(file.type)) {
    showError("Invalid file type. Please use PNG, JPG, GIF, or WEBP.");
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError("File too large. Maximum size is 16 MB.");
    return;
  }

  selectedFile = file;

  /* Preview */
  const url = URL.createObjectURL(file);
  imgPreview.src = url;

  /* Size badge */
  imgBadge.textContent = formatFileSize(file.size);

  /* Source tag: shows "📋 Pasted", "📁 Uploaded", or "⬇ Dropped" */
  const sourceLabels = {
    paste:  "📋 Pasted",
    upload: "📁 Uploaded",
    drop:   "⬇ Dropped",
  };
  imgSourceTag.textContent = sourceLabels[source] || "📁 Uploaded";

  /* Show preview, hide drop zone */
  dropZone.hidden       = true;
  imgPreviewWrap.hidden = false;

  refreshAnalyzeBtn();
}

function clearFile() {
  selectedFile          = null;
  imgPreview.src        = "";
  imgPreviewWrap.hidden = true;
  dropZone.hidden       = false;
  fileInput.value       = "";
  imgSourceTag.textContent = "";
  refreshAnalyzeBtn();
}

/* ═══════════════════════════════════════════════════════════════
   1. CLICK TO BROWSE
   ═══════════════════════════════════════════════════════════════ */
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) setFile(fileInput.files[0], "upload");
});

/* ═══════════════════════════════════════════════════════════════
   2. DRAG & DROP
   ═══════════════════════════════════════════════════════════════ */
["dragenter","dragover"].forEach(evt =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragging");
  })
);
["dragleave","dragend"].forEach(evt =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove("dragging"))
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragging");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file, "drop");
});

/* ═══════════════════════════════════════════════════════════════
   3. PASTE  (Ctrl+V / Cmd+V)
   ═══════════════════════════════════════════════════════════════ */

/**
 * Listen for paste events anywhere on the page.
 * If the clipboard contains an image item, grab it as a File.
 * If it's text only, ignore (let the textarea handle it normally).
 */
document.addEventListener("paste", (e) => {
  /* Don't intercept if user is actively typing in the textarea */
  if (document.activeElement === memeText) return;

  const items = Array.from(e.clipboardData?.items || []);
  const imageItem = items.find(item => item.type.startsWith("image/"));

  if (!imageItem) return;   /* no image in clipboard — ignore */

  e.preventDefault();

  const file = imageItem.getAsFile();
  if (!file) return;

  /* Flash drop zone for feedback */
  dropZone.classList.add("paste-active");
  setTimeout(() => dropZone.classList.remove("paste-active"), 600);

  setFile(file, "paste");
});

/* Also handle paste directly on the drop zone div */
dropZone.addEventListener("paste", (e) => {
  const items = Array.from(e.clipboardData?.items || []);
  const imageItem = items.find(item => item.type.startsWith("image/"));
  if (!imageItem) return;
  e.preventDefault();
  const file = imageItem.getAsFile();
  if (file) setFile(file, "paste");
});

/* Remove image button */
removeImgBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearFile();
});

/* ═══════════════════════════════════════════════════════════════
   TEXTAREA — char counter
   ═══════════════════════════════════════════════════════════════ */
memeText.addEventListener("input", () => {
  const len = memeText.value.length;
  charCount.textContent = `${len} / 512`;
  refreshAnalyzeBtn();
});

/* ═══════════════════════════════════════════════════════════════
   BUTTON STATE
   ═══════════════════════════════════════════════════════════════ */
function refreshAnalyzeBtn() {
  analyzeBtn.disabled = !(selectedFile && memeText.value.trim());
}

/* ═══════════════════════════════════════════════════════════════
   ANALYSE
   ═══════════════════════════════════════════════════════════════ */
analyzeBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (!selectedFile || !memeText.value.trim()) return;

  setUiState("loading");
  startLoadingSteps();

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("text", memeText.value.trim());

  try {
    const resp = await fetch("/predict", { method: "POST", body: formData });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || `Server error ${resp.status}`);
    }

    const data = await resp.json();
    if (!data.success) throw new Error(data.error || "Prediction failed.");

    clearLoadingSteps();
    renderResult(data);
    setUiState("result");

  } catch (err) {
    clearLoadingSteps();
    showError(err.message);
  }
}

/* ── Loading animation ───────────────────────────────────────── */
function startLoadingSteps() {
  lsItems.forEach(el => el.classList.remove("active","done"));
  lsItems[0].classList.add("active");
  let step = 0;
  loadingTimer = setInterval(() => {
    lsItems[step].classList.replace("active","done");
    step++;
    if (step < lsItems.length) lsItems[step].classList.add("active");
    else clearInterval(loadingTimer);
  }, 700);
}

function clearLoadingSteps() {
  clearInterval(loadingTimer);
}

/* ── Render result ───────────────────────────────────────────── */
function renderResult(data) {
  const isHateful = data.label_id === 1;

  /* Verdict banner */
  verdictBanner.className = `verdict-banner ${isHateful ? "hateful" : "safe"}`;
  verdictIcon.textContent  = isHateful ? "🚨" : "✅";
  verdictLabel.textContent = data.label;
  verdictSub.textContent   = isHateful
    ? "This meme contains potentially hateful content."
    : "No hateful content detected in this meme.";

  /* Confidence bar — animate after paint */
  confPct.textContent       = `${data.confidence}%`;
  confBarFill.style.width   = "0%";
  requestAnimationFrame(() => { confBarFill.style.width = `${data.confidence}%`; });

  /* Probability bars */
  hatefulFill.style.width    = "0%";
  notHatefulFill.style.width = "0%";
  hatefulPct.textContent     = `${data.hateful_prob}%`;
  notHatefulPct.textContent  = `${data.not_hateful_prob}%`;
  requestAnimationFrame(() => {
    hatefulFill.style.width    = `${data.hateful_prob}%`;
    notHatefulFill.style.width = `${data.not_hateful_prob}%`;
  });

  /* Thumbnail */
  if (data.image_url) {
    resultThumb.src        = data.image_url;
    resultThumbWrap.hidden = false;
  } else {
    resultThumbWrap.hidden = true;
  }
}

/* ═══════════════════════════════════════════════════════════════
   UI STATE MACHINE
   ═══════════════════════════════════════════════════════════════ */
function setUiState(state) {
  resultIdle.hidden    = state !== "idle";
  resultLoading.hidden = state !== "loading";
  resultOutput.hidden  = state !== "result";
  resultError.hidden   = state !== "error";

  const btnText    = analyzeBtn.querySelector(".btn-text");
  const btnSpinner = analyzeBtn.querySelector(".btn-spinner");

  if (state === "loading") {
    analyzeBtn.disabled = true;
    btnText.hidden      = true;
    btnSpinner.hidden   = false;
  } else {
    btnText.hidden    = false;
    btnSpinner.hidden = true;
    refreshAnalyzeBtn();
  }
}

/* ── Error ───────────────────────────────────────────────────── */
function showError(msg) {
  errorMsg.textContent = msg;
  setUiState("error");
}
errorRetryBtn.addEventListener("click", () => setUiState("idle"));

/* ── Reset ───────────────────────────────────────────────────── */
resetBtn.addEventListener("click", fullReset);

function fullReset() {
  clearFile();
  memeText.value        = "";
  charCount.textContent = "0 / 512";
  setUiState("idle");
  clearLoadingSteps();
}

/* ═══════════════════════════════════════════════════════════════
   UTILITIES
   ═══════════════════════════════════════════════════════════════ */
function formatFileSize(bytes) {
  if (bytes < 1024)        return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/* ── Init ────────────────────────────────────────────────────── */
setUiState("idle");
