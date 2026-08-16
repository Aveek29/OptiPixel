const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const uploadSection = document.getElementById("upload-section");
const processingSection = document.getElementById("processing-section");
const resultSection = document.getElementById("result-section");
const errorSection = document.getElementById("error-section");
const statusText = document.getElementById("status-text");
const progressFill = document.getElementById("progress-fill");
const errorText = document.getElementById("error-text");

const funMessages = [
    "AI is squinting at your pixels...",
    "Teaching neural networks to see better...",
    "Asking the computer to imagine higher resolution...",
    "Reconstructing details that never existed...",
    "Applying magic — the mathematical kind...",
    "Telling apart noise from actual content...",
    "Your image is getting a PhD in clarity...",
    "Running Real-ESRGAN — sounds cooler than it feels on CPU...",
    "Converting blurry dreams into sharp reality...",
    "The model is thinking very hard right now...",
    "Enhancing. This is what AI was made for.",
    "Every pixel matters. Processing them all.",
    "Super-resolution isn't super fast on CPU. Hold on.",
    "Your image is in the gym. Getting gains.",
    "Almost like Photoshop, but with math.",
    "Real-ESRGAN says hello. It's working.",
    "Convolutions happen. Details emerge.",
    "This would be instant on a GPU. Blame the CPU.",
    "Denoising. Sharpening. Upscaling. Repeat.",
    "The AI promises it's worth the wait.",
];

let funInterval = null;

function startFunMessages() {
    let idx = 0;
    statusText.textContent = funMessages[0];
    funInterval = setInterval(() => {
        idx = (idx + 1) % funMessages.length;
        statusText.textContent = funMessages[idx];
    }, 3000);
}

function stopFunMessages() {
    if (funInterval) {
        clearInterval(funInterval);
        funInterval = null;
    }
}

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => { dropZone.classList.remove("dragover"); });
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length > 0) handleFile(fileInput.files[0]); });

async function handleFile(file) {
    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (!allowed.includes(file.type)) {
        showError("Unsupported file type. Please use JPG, PNG, or WebP.");
        return;
    }
    const maxMb = parseInt(document.getElementById("max-upload").textContent) || 10;
    if (file.size > maxMb * 1024 * 1024) {
        showError(`File exceeds ${maxMb}MB limit.`);
        return;
    }
    showProcessing();
    const formData = new FormData();
    formData.append("file", file);

    try {
        setStep("analyze", "active");
        setProgress(15);
        startFunMessages();

        const resp = await fetch("/api/enhance", { method: "POST", body: formData });

        stopFunMessages();
        setStep("analyze", "done");
        setProgress(60);
        setStep("enhance", "active");
        statusText.textContent = "Almost there — finalizing enhancement...";

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: "Enhancement failed" }));
            if (resp.status === 504) throw new Error("Processing took too long (3min limit). Try a smaller image.");
            throw new Error(err.detail || `Server error: ${resp.status}`);
        }

        const data = await resp.json();
        setStep("enhance", "done");
        setProgress(90);
        setStep("ready", "active");
        statusText.textContent = "Done! Preparing download...";
        setProgress(100);
        setTimeout(() => showResult(data, file), 400);
    } catch (err) {
        stopFunMessages();
        showError(err.message || "Enhancement failed. Please try again.");
    }
}

function showProcessing() {
    uploadSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    errorSection.classList.add("hidden");
    processingSection.classList.remove("hidden");
    resetSteps();
    setProgress(0);
    setStep("upload", "done");
    setProgress(5);
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function showResult(data, originalFile) {
    processingSection.classList.add("hidden");
    errorSection.classList.add("hidden");
    resultSection.classList.remove("hidden");

    const origImg = document.getElementById("original-preview");
    if (originalFile) origImg.src = URL.createObjectURL(originalFile);

    const enhImg = document.getElementById("enhanced-preview");
    if (data.download_url) enhImg.src = data.download_url;

    const dlBtn = document.getElementById("download-btn");
    if (data.download_url) {
        dlBtn.href = data.download_url;
        dlBtn.download = "optic-pixel-enhanced.png";
    }

    if (data.analysis) {
        const a = data.analysis;
        document.getElementById("original-meta").innerHTML =
            `${a.width}&times;${a.height} &bull; ${a.fmt} &bull; ${formatBytes(a.file_size_bytes)}<br>` +
            `Sharpness: ${a.sharpness_score} &bull; Brightness: ${a.mean_brightness}<br>` +
            `Profile: ${a.profile} &mdash; ${a.profile_description}`;
    }

    const scaleText = data.scale ? `${data.scale}x` : "1x";
    document.getElementById("enhanced-meta").innerHTML =
        `${data.output_width}&times;${data.output_height} &bull; ${formatBytes(data.output_size_bytes)}<br>` +
        `Scale: ${scaleText} &bull; ${data.engine}`;

    const timeSec = (data.processing_ms / 1000).toFixed(1);
    document.getElementById("result-stats").innerHTML =
        `<strong>Processing time:</strong> ${timeSec}s &bull; ` +
        `<strong>Job:</strong> ${data.job_id}<br>` +
        `<strong>Input:</strong> ${data.analysis ? data.analysis.width + '&times;' + data.analysis.height : '?'} (${formatBytes(data.analysis ? data.analysis.file_size_bytes : 0)}) &rarr; ` +
        `<strong>Output:</strong> ${data.output_width}&times;${data.output_height} (${formatBytes(data.output_size_bytes)})<br>` +
        `<strong>Scale:</strong> ${scaleText} &bull; ` +
        `<strong>Profile:</strong> ${data.analysis ? data.analysis.profile : '?'} &bull; ` +
        `<strong>Engine:</strong> ${data.engine}`;
}

function showError(msg) {
    uploadSection.classList.add("hidden");
    processingSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    errorSection.classList.remove("hidden");
    errorText.textContent = msg;
}

function resetUI() {
    uploadSection.classList.remove("hidden");
    processingSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    errorSection.classList.add("hidden");
    fileInput.value = "";
    stopFunMessages();
    resetSteps();
    setProgress(0);
}

function setProgress(pct) { progressFill.style.width = pct + "%"; }
function setStatus(text) { statusText.textContent = text; }
function setStep(name, state) {
    const el = document.getElementById("step-" + name);
    if (el) { el.classList.remove("active", "done"); if (state) el.classList.add(state); }
}
function resetSteps() { ["upload", "analyze", "enhance", "ready"].forEach((s) => setStep(s, "")); }
