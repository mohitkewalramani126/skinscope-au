// SkinScope AU — frontend interaction logic.
// Ported from the Claude Design export (dc-runtime component) to plain,
// dependency-free JS. Calls POST /api/analyze per agent/schemas.py's
// AgentResponse contract; falls back to a clearly-labelled simulated
// response if the backend is unreachable.

(function () {
  "use strict";

  // ---------- element refs ----------
  const dropzone = document.getElementById("ss-dropzone");
  const fileInput = document.getElementById("ss-file");
  const previewImg = document.getElementById("ss-preview-img");
  const placeholder = document.getElementById("ss-dropzone-placeholder");
  const clearBtn = document.getElementById("ss-clear-btn");
  const sampleABtn = document.getElementById("ss-sample-a");
  const sampleBBtn = document.getElementById("ss-sample-b");
  const questionInput = document.getElementById("ss-q");
  const sensitivityCheckbox = document.getElementById("ss-sensitivity-checkbox");
  const analyzeBtn = document.getElementById("ss-analyze-btn");
  const hintEl = document.getElementById("ss-hint");
  const loadingEl = document.getElementById("ss-loading");
  const resultsEl = document.getElementById("ss-results");
  const figureEl = document.getElementById("ss-figure");
  const maskedImg = document.getElementById("ss-masked-img");
  const demoBadge = document.getElementById("ss-demo-badge");
  const badgeEl = document.getElementById("ss-badge");
  const scoreTextEl = document.getElementById("ss-score-text");
  const bandLabelEl = document.getElementById("ss-band-label");
  const alertEl = document.getElementById("ss-alert");
  const alertReasonEl = document.getElementById("ss-alert-reason");
  const answerWrap = document.getElementById("ss-answer-wrap");
  const answerTextEl = document.getElementById("ss-answer-text");
  const citesWrap = document.getElementById("ss-cites-wrap");
  const citesList = document.getElementById("ss-cites-list");
  const disclaimerEcho = document.getElementById("ss-disclaimer-echo");
  const infoToggle = document.getElementById("info-toggle");
  const infoPanelWrap = document.getElementById("info-panel-wrap");
  const infoLabel = document.getElementById("info-toggle-label");
  const infoChev = document.getElementById("info-toggle-chev");

  let currentFile = null; // Blob/File to upload
  let sampleBand = null;  // "low" | "high" | null — hints the simulate() fallback

  function reducedMotion() {
    return matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  // ---------- acknowledgment gate (Day 14: public demo requirement) ----------
  // Blocks interaction until the visitor acknowledges this isn't a diagnosis.
  // Persisted in localStorage so it only interrupts once per browser, not
  // every single page load -- the disclaimer banner stays up permanently
  // regardless, this gate is just the one-time "you must actively agree" step.
  (function initAcknowledgmentGate() {
    const gate = document.getElementById("ss-gate");
    const acceptBtn = document.getElementById("ss-gate-accept");
    if (!gate || !acceptBtn) return;

    const ACK_KEY = "ss_acknowledged_v1";
    let acknowledged = false;
    try {
      acknowledged = localStorage.getItem(ACK_KEY) === "1";
    } catch (e) {
      // localStorage unavailable (private browsing, disabled storage, etc.)
      // -- fail open to showing the gate every time rather than crashing.
    }

    if (acknowledged) {
      gate.hidden = true;
      return;
    }

    document.body.style.overflow = "hidden";
    acceptBtn.addEventListener("click", () => {
      gate.hidden = true;
      document.body.style.overflow = "";
      try {
        localStorage.setItem(ACK_KEY, "1");
      } catch (e) {
        // ignore -- worst case the gate reappears next visit
      }
    });
  })();

  // ---------- info toggle ----------
  let infoOpen = true;
  infoToggle.addEventListener("click", () => {
    infoOpen = !infoOpen;
    infoToggle.setAttribute("aria-expanded", String(infoOpen));
    infoPanelWrap.classList.toggle("ss-collapsed", !infoOpen);
    infoLabel.textContent = infoOpen ? "Hide" : "Show";
    infoChev.style.transform = infoOpen ? "rotate(180deg)" : "rotate(0deg)";
  });

  // ---------- 3D tilt on hover cards ----------
  document.querySelectorAll(".ss-tilt").forEach((el) => {
    el.addEventListener("mousemove", (e) => {
      if (reducedMotion()) return;
      const r = el.getBoundingClientRect();
      const px = (e.clientX - r.left) / r.width - 0.5;
      const py = (e.clientY - r.top) / r.height - 0.5;
      el.style.transform = `perspective(900px) rotateX(${(-py * 4).toFixed(2)}deg) rotateY(${(px * 4).toFixed(2)}deg) translateY(-4px)`;
      el.style.boxShadow = "0 30px 60px -30px rgba(50,38,26,.55)";
    });
    el.addEventListener("mouseleave", () => {
      el.style.transform = "none";
      el.style.boxShadow = "";
    });
  });

  // ---------- reveal-on-scroll (hero + workspace card) ----------
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const delay = entry.target.getAttribute("data-reveal") || 0;
          entry.target.style.transitionDelay = delay + "ms";
          entry.target.classList.add("ss-revealed");
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.14 }
  );
  document.querySelectorAll("[data-reveal]").forEach((el) => io.observe(el));

  // ---------- image loading helpers ----------
  function setImage(dataUrl) {
    previewImg.src = dataUrl;
    previewImg.hidden = false;
    placeholder.hidden = true;
    clearBtn.hidden = false;
  }
  function clearImage() {
    currentFile = null;
    sampleBand = null;
    previewImg.hidden = true;
    previewImg.src = "";
    placeholder.hidden = false;
    clearBtn.hidden = true;
  }
  function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  // ---------- drag & drop / file picker ----------
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("ss-dz-active"); });
  dropzone.addEventListener("dragleave", (e) => { e.preventDefault(); dropzone.classList.remove("ss-dz-active"); });
  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("ss-dz-active");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f && f.type.indexOf("image/") === 0) {
      currentFile = f; sampleBand = null;
      setImage(await readFileAsDataUrl(f));
    }
  });
  fileInput.addEventListener("change", async (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) {
      currentFile = f; sampleBand = null;
      setImage(await readFileAsDataUrl(f));
    }
  });
  clearBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    clearImage();
  });

  // ---------- sample images (real project test photos, not synthetic) ----------
  async function loadSample(url, band) {
    const resp = await fetch(url);
    const blob = await resp.blob();
    currentFile = blob;
    sampleBand = band;
    setImage(await readFileAsDataUrl(blob));
  }
  sampleABtn.addEventListener("click", (e) => { e.preventDefault(); loadSample("images/sample-a.jpg", "low"); });
  sampleBBtn.addEventListener("click", (e) => { e.preventDefault(); loadSample("images/sample-b.jpg", "high"); });

  // ---------- example question buttons ----------
  document.querySelectorAll("[data-example]").forEach((btn) => {
    btn.addEventListener("click", () => { questionInput.value = btn.getAttribute("data-example"); });
  });

  // ---------- canvas compositing ----------
  function loadImg(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }
  function bandHex(band) {
    return band === "high" ? "#A85B41" : band === "moderate" ? "#9A7326" : "#2F7A67";
  }
  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  async function composite(imgSrc, maskB64, band) {
    try {
      const base = await loadImg(imgSrc);
      const mask = await loadImg("data:image/png;base64," + maskB64);
      const w = base.naturalWidth, h = base.naturalHeight;
      const c = document.createElement("canvas"); c.width = w; c.height = h;
      const ctx = c.getContext("2d");
      ctx.drawImage(base, 0, 0, w, h);
      const t = document.createElement("canvas"); t.width = w; t.height = h;
      const tctx = t.getContext("2d");
      tctx.drawImage(mask, 0, 0, w, h);
      tctx.globalCompositeOperation = "source-in";
      tctx.fillStyle = bandHex(band);
      tctx.fillRect(0, 0, w, h);
      ctx.globalAlpha = 0.5; ctx.drawImage(t, 0, 0); ctx.globalAlpha = 1;
      return c.toDataURL("image/png");
    } catch (e) {
      return imgSrc;
    }
  }
  async function demoComposite(imgSrc, band) {
    try {
      const base = await loadImg(imgSrc);
      const w = base.naturalWidth, h = base.naturalHeight;
      const c = document.createElement("canvas"); c.width = w; c.height = h;
      const ctx = c.getContext("2d");
      ctx.drawImage(base, 0, 0, w, h);
      const cx = w / 2, cy = h / 2, r = Math.min(w, h) * 0.34;
      const col = bandHex(band);
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      g.addColorStop(0, hexA(col, 0.55)); g.addColorStop(0.62, hexA(col, 0.30)); g.addColorStop(1, hexA(col, 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.ellipse(cx, cy, r, r * 0.86, 0, 0, 6.2832); ctx.fill();
      ctx.strokeStyle = hexA(col, 0.8); ctx.lineWidth = Math.max(2, w * 0.006);
      ctx.setLineDash([w * 0.02, w * 0.02]);
      ctx.beginPath(); ctx.ellipse(cx, cy, r * 0.8, r * 0.7, 0, 0, 6.2832); ctx.stroke();
      return c.toDataURL("image/png");
    } catch (e) {
      return imgSrc;
    }
  }

  // ---------- fallback simulated response (only used if the backend call fails) ----------
  // Citations here are restricted to sources actually verified during Day 10's
  // corpus research (rag/corpus.json) -- not invented URLs.
  // sensitivityMode mirrors agent/nodes.py's safety_gate_node: "standard" only
  // escalates on "high" band, "high" also escalates on "moderate" -- kept in
  // sync here so the demo fallback behaves the same as the real backend.
  function simulate(hasImage, question, sensitivityMode) {
    const disclaimer = "SkinScope AU is an awareness tool, not a medical device. It does not diagnose skin cancer, and a low-risk result does not rule it out. Always have any new, changing, or concerning spot checked by a doctor or dermatologist.";
    const diag = /\b(do i have|is (this|it) (cancer|melanoma|malignant)|is it cancer|am i (fine|ok|okay)|diagnos|is this spot dangerous)\b/i;
    if (question && diag.test(question)) {
      return {
        mask_png_base64: null, risk_score: null, risk_band: null, refused: true, escalate: false, escalation_reason: null,
        answer: "I can't tell you whether a specific spot is or isn't skin cancer. That's a diagnosis only a qualified doctor can make, and getting it wrong in either direction carries real risk. What I can do is help you notice the kinds of changes worth acting on. If this spot is new, changing, itching, bleeding, or simply worrying you, please book a GP or dermatologist appointment to have it examined in person.",
        citations: [], disclaimer,
      };
    }
    const band = sampleBand || (hasImage ? (Math.random() < 0.55 ? "low" : "moderate") : null);
    const score = band === "high" ? 0.71 : band === "moderate" ? 0.39 : band === "low" ? 0.13 : null;
    const escalate = sensitivityMode === "high" ? (band === "high" || band === "moderate") : band === "high";
    const citations = [
      { title: "Cancer Council Australia: Check for signs of skin cancer", url: "https://www.cancer.org.au/cancer-information/causes-and-prevention/sun-safety/check-for-signs-of-skin-cancer" },
      { title: "healthdirect: Should I be checked for skin cancer?", url: "https://www.healthdirect.gov.au/should-I-be-checked-for-skin-cancer" },
      { title: "Melanoma Institute Australia: Checking Your Skin", url: "https://melanoma.org.au/about-melanoma/checking-your-skin/" },
    ];
    let answer;
    if (!hasImage && question) {
      answer = "When you check your own skin, the ABCDE guide is a useful start: Asymmetry, Border irregularity, Colour variation, Diameter over about 6mm, and Evolving or changing spots. Do a full-body check monthly in good light, photograph anything you want to track, and see a doctor about anything new, changing, or concerning. This tool can prompt you, but it can't judge an individual spot for you.";
    } else if (band === "high") {
      answer = "This spot shows features the model flags as higher-signal. Remember the model is only estimating and this is not a diagnosis, but given the result, the responsible next step is to have it looked at by a GP or dermatologist soon. Note its size, colour and any change over time, and mention anything that itches or bleeds.";
    } else {
      answer = "The model reads this spot as lower-signal, but that does not rule anything out. At this operating point it misses about half of malignant cases, and it is less sensitive on darker skin tones. Keep an eye on it, photograph it monthly to compare, and see a doctor if it changes in size, shape or colour, or starts to itch or bleed.";
    }
    let escalationReason = null;
    if (escalate) {
      escalationReason = band === "moderate"
        ? "You've enabled high-sensitivity mode, which also flags moderate-signal results for an in-person check. At the standard setting, this tool misses about half of real cases; this mode catches more of them at the cost of more false alarms. Please see a doctor or dermatologist for a proper assessment. Do not rely on this tool's output alone."
        : "This result is higher-signal. Please have this spot reviewed in person by a GP or dermatologist, sooner rather than later.";
    }
    return {
      mask_png_base64: null, risk_score: score, risk_band: band, refused: false, escalate,
      escalation_reason: escalationReason,
      answer, citations, disclaimer,
    };
  }

  // ---------- count-up animation for the risk score ----------
  let countRaf = null;
  function countUp(target) {
    if (target == null) return;
    if (reducedMotion()) { scoreTextEl.textContent = Math.round(target * 100); return; }
    const start = performance.now(), dur = 950;
    function step(now) {
      const p = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      scoreTextEl.textContent = Math.round(target * 100 * eased);
      if (p < 1) countRaf = requestAnimationFrame(step);
    }
    countRaf = requestAnimationFrame(step);
  }

  // ---------- reveal results with staggered delays ----------
  function revealResult(el, delayMs) {
    el.style.transitionDelay = delayMs + "ms";
    requestAnimationFrame(() => el.classList.add("ss-revealed"));
  }

  // ---------- main analyze flow ----------
  analyzeBtn.addEventListener("click", async () => {
    const question = questionInput.value.trim();
    if (!currentFile && !question) {
      hintEl.textContent = "Add a photo or type a question to begin. At least one is needed.";
      hintEl.hidden = false;
      return;
    }
    hintEl.hidden = true;
    if (countRaf) cancelAnimationFrame(countRaf);

    analyzeBtn.disabled = true;
    resultsEl.hidden = true;
    loadingEl.hidden = false;
    // reset reveal state on any elements we're about to reuse
    [figureEl, badgeEl, alertEl, answerWrap, citesWrap, disclaimerEcho].forEach((el) => el.classList.remove("ss-revealed"));

    const sensitivityMode = sensitivityCheckbox && sensitivityCheckbox.checked ? "high" : "standard";

    let res, demo = false;
    try {
      const fd = new FormData();
      if (currentFile) fd.append("file", currentFile, "upload.png");
      if (question) fd.append("question", question);
      fd.append("sensitivity_mode", sensitivityMode);
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      if (!r.ok) throw new Error("bad status");
      res = await r.json();
    } catch (e) {
      demo = true;
      res = simulate(!!currentFile, question, sensitivityMode);
      await new Promise((resolve) => setTimeout(resolve, 1700));
    }

    const imgSrc = previewImg.hidden ? null : previewImg.src;
    let masked = null;
    if (res.mask_png_base64 && imgSrc) {
      masked = await composite(imgSrc, res.mask_png_base64, res.risk_band);
    } else if (demo && imgSrc && res.risk_band) {
      masked = await demoComposite(imgSrc, res.risk_band);
    }

    loadingEl.hidden = true;
    analyzeBtn.disabled = false;
    resultsEl.hidden = false;

    const band = res.risk_band;
    const bandFg = bandHex(band);
    const bandLabelText = band === "high" ? "Higher signal" : band === "moderate" ? "Moderate signal" : band === "low" ? "Low signal" : "N/A";

    const showFigure = !!masked;
    const showBadge = res.risk_score != null && !!band;
    const showAlert = !!res.escalate;
    const showCites = !res.refused && res.citations && res.citations.length > 0;

    resultsEl.classList.toggle("ss-has-figure", showFigure);

    figureEl.hidden = !showFigure;
    if (showFigure) maskedImg.src = masked;

    demoBadge.hidden = !demo;

    badgeEl.hidden = !showBadge;
    if (showBadge) {
      badgeEl.style.setProperty("--risk-color", bandFg);
      badgeEl.style.setProperty("--risk-tint", hexA(bandFg, 0.12));
      bandLabelEl.textContent = bandLabelText;
      scoreTextEl.textContent = "0";
    }

    alertEl.hidden = !showAlert;
    if (showAlert) alertReasonEl.textContent = res.escalation_reason || "";

    answerWrap.hidden = !res.answer;
    if (res.answer) answerTextEl.textContent = res.answer;

    citesWrap.hidden = !showCites;
    if (showCites) {
      citesList.innerHTML = "";
      res.citations.forEach((c) => {
        const a = document.createElement("a");
        a.href = c.url; a.target = "_blank"; a.rel = "noopener noreferrer"; a.className = "ss-cite-link";
        a.innerHTML = '<span aria-hidden="true" class="ss-cite-arrow">&#8599;</span><span></span>';
        a.querySelector("span:last-child").textContent = c.title;
        citesList.appendChild(a);
      });
    }

    disclaimerEcho.textContent = res.disclaimer || "";

    requestAnimationFrame(() => {
      revealResult(figureEl, 0);
      revealResult(badgeEl, 140);
      revealResult(alertEl, 120);
      revealResult(answerWrap, 300);
      revealResult(citesWrap, 460);
      revealResult(disclaimerEcho, 560);
      if (showBadge) countUp(res.risk_score);
    });
  });
})();
