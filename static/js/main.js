// JavaScript for Interactive Bayesian Spam Image Filter with Instant Live Sliders

document.addEventListener("DOMContentLoaded", function () {
    // Sliders & Displays
    const strengthRange = document.getElementById("strengthRange");
    const strengthDisplay = document.getElementById("strengthDisplay");
    
    const balanceRange = document.getElementById("balanceRange");
    const balanceDisplay = document.getElementById("balanceDisplay");
    
    const priorRange = document.getElementById("priorRange");
    const priorDisplay = document.getElementById("priorDisplay");
    
    const thresholdRange = document.getElementById("thresholdRange");
    const thresholdDisplay = document.getElementById("thresholdDisplay");
    
    const blendRange = document.getElementById("blendRange");
    const blendDisplay = document.getElementById("blendDisplay");
    
    const filterType = document.getElementById("filterType");
    const filterStrengthRange = document.getElementById("filterStrengthRange");
    const filterStrengthDisplay = document.getElementById("filterStrengthDisplay");

    // Inputs & Form
    const dropZone = document.getElementById("dropZone");
    const imageInput = document.getElementById("imageInput");
    const selectedFileName = document.getElementById("selectedFileName");
    const samplePathInput = document.getElementById("samplePath");
    const cachedFilenameInput = document.getElementById("cachedFilename");
    const classifyForm = document.getElementById("classifyForm");

    // State Elements
    const placeholderState = document.getElementById("placeholderState");
    const loadingState = document.getElementById("loadingState");
    const resultState = document.getElementById("resultState");

    // Result Card Elements
    const verdictCard = document.getElementById("verdictCard");
    const verdictIconContainer = document.getElementById("verdictIconContainer");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictText = document.getElementById("verdictText");
    const confidenceText = document.getElementById("confidenceText");
    const elapsedTime = document.getElementById("elapsedTime");

    const spamProbText = document.getElementById("spamProbText");
    const hamProbText = document.getElementById("hamProbText");
    const logOddsText = document.getElementById("logOddsText");
    const colorLLRText = document.getElementById("colorLLRText");
    const edgeLLRText = document.getElementById("edgeLLRText");
    const lapLLRText = document.getElementById("lapLLRText");

    // Images
    const imgOrig = document.getElementById("imgOrig");
    const imgHeat = document.getElementById("imgHeat");
    const imgOverlay = document.getElementById("imgOverlay");
    const imgDiag = document.getElementById("imgDiag");

    let isRequestInProgress = false;
    let pendingRequest = false;
    let debounceTimer = null;

    // Helper: Trigger live update with 60ms debounce for buttery-smooth interaction
    function triggerLiveUpdate() {
        const hasFile = imageInput.files && imageInput.files.length > 0;
        const hasSample = samplePathInput.value !== "";
        const hasCached = cachedFilenameInput.value !== "";

        if (!hasFile && !hasSample && !hasCached) {
            return; // No image loaded yet
        }

        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            executeClassification(true);
        }, 60);
    }

    // --- Slider Listeners (Live input events) ---

    if (strengthRange && strengthDisplay) {
        strengthRange.addEventListener("input", (e) => {
            strengthDisplay.textContent = parseFloat(e.target.value).toFixed(1) + "x";
            triggerLiveUpdate();
        });
    }

    if (balanceRange && balanceDisplay) {
        balanceRange.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            const colorPct = Math.round((1 - val) * 100);
            const edgePct = Math.round(val * 100);
            balanceDisplay.textContent = `${colorPct}% Color / ${edgePct}% Edge`;
            triggerLiveUpdate();
        });
    }

    if (priorRange && priorDisplay) {
        priorRange.addEventListener("input", (e) => {
            priorDisplay.textContent = parseFloat(e.target.value).toFixed(2);
            triggerLiveUpdate();
        });
    }

    if (thresholdRange && thresholdDisplay) {
        thresholdRange.addEventListener("input", (e) => {
            thresholdDisplay.textContent = parseFloat(e.target.value).toFixed(2);
            triggerLiveUpdate();
        });
    }

    if (blendRange && blendDisplay) {
        blendRange.addEventListener("input", (e) => {
            const pct = Math.round(parseFloat(e.target.value) * 100);
            blendDisplay.textContent = `${pct}%`;
            triggerLiveUpdate();
        });
    }

    if (filterType) {
        filterType.addEventListener("change", () => {
            triggerLiveUpdate();
        });
    }

    if (filterStrengthRange && filterStrengthDisplay) {
        filterStrengthRange.addEventListener("input", (e) => {
            filterStrengthDisplay.textContent = parseFloat(e.target.value).toFixed(1) + "x";
            triggerLiveUpdate();
        });
    }

    // --- Drag and Drop File Handlers ---

    if (dropZone) {
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add("dragover");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove("dragover");
            }, false);
        });

        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                imageInput.files = files;
                handleFileSelection(files[0]);
            }
        });
    }

    if (imageInput) {
        imageInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    function handleFileSelection(file) {
        selectedFileName.textContent = `Selected: ${file.name}`;
        samplePathInput.value = "";
        cachedFilenameInput.value = ""; // Reset cached file for new upload
        document.querySelectorAll(".sample-thumb").forEach(el => {
            el.classList.remove("selected", "selected-ham");
        });
        // Run classification immediately on file select
        executeClassification(false);
    }

    // --- Form Submission / Execution ---

    if (classifyForm) {
        classifyForm.addEventListener("submit", function (e) {
            e.preventDefault();
            executeClassification(false);
        });
    }

    async function executeClassification(isLiveSliderUpdate = false) {
        const hasFile = imageInput.files && imageInput.files.length > 0;
        const hasSample = samplePathInput.value !== "";
        const hasCached = cachedFilenameInput.value !== "";

        if (!hasFile && !hasSample && !hasCached) {
            alert("Please select or upload an image first!");
            return;
        }

        if (isRequestInProgress) {
            pendingRequest = true;
            return;
        }

        isRequestInProgress = true;

        if (!isLiveSliderUpdate) {
            placeholderState.classList.add("d-none");
            resultState.classList.add("d-none");
            loadingState.classList.remove("d-none");
        }

        const formData = new FormData(classifyForm);

        // If we already have a cached server filename and are not uploading a fresh file,
        // delete the binary file from formData to keep requests ultra-lightweight (<1ms network)
        if (cachedFilenameInput.value !== "" && isLiveSliderUpdate) {
            formData.delete("image");
        }

        try {
            const response = await fetch("/classify", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            loadingState.classList.add("d-none");

            if (!response.ok || data.error) {
                if (!isLiveSliderUpdate) {
                    alert("Error: " + (data.error || "Failed to classify image."));
                    placeholderState.classList.remove("d-none");
                }
                isRequestInProgress = false;
                return;
            }

            // Save cached filename for rapid subsequent slider adjustments
            if (data.cached_filename) {
                cachedFilenameInput.value = data.cached_filename;
            }

            // Render Results
            resultState.classList.remove("d-none");

            const isSpam = data.prediction === "SPAM";

            // Verdict Styling
            verdictCard.className = "card shadow-sm border-0 rounded-3 mb-3 " + (isSpam ? "is-spam" : "is-ham");
            verdictIconContainer.className = "p-3 rounded-circle me-3 " + (isSpam ? "is-spam" : "is-ham");
            verdictIcon.className = "fa-solid fs-2 " + (isSpam ? "fa-triangle-exclamation" : "fa-shield-check");
            verdictText.textContent = isSpam ? "SPAM DETECTED" : "HAM (LEGITIMATE)";
            verdictText.className = "mb-0 fw-black " + (isSpam ? "text-danger" : "text-success");

            confidenceText.textContent = `${data.confidence}%`;
            elapsedTime.textContent = `${data.elapsed_ms} ms`;

            spamProbText.textContent = `${data.spam_probability}%`;
            hamProbText.textContent = `${data.ham_probability}%`;
            logOddsText.textContent = (data.log_odds >= 0 ? "+" : "") + data.log_odds.toFixed(2);

            // Sub-filter details
            if (colorLLRText) colorLLRText.textContent = (data.color_llr >= 0 ? "+" : "") + data.color_llr.toFixed(2);
            if (edgeLLRText) edgeLLRText.textContent = (data.edge_llr >= 0 ? "+" : "") + data.edge_llr.toFixed(2);
            if (lapLLRText) lapLLRText.textContent = (data.lap_llr >= 0 ? "+" : "") + data.lap_llr.toFixed(2);

            // Update Image Previews (with cache-busting timestamp)
            const ts = new Date().getTime();
            imgOrig.src = data.image_orig_url + "?t=" + ts;
            imgHeat.src = data.image_heat_url + "?t=" + ts;
            imgOverlay.src = data.image_overlay_url + "?t=" + ts;
            imgDiag.src = data.image_diag_url + "?t=" + ts;

        } catch (err) {
            console.error(err);
            if (!isLiveSliderUpdate) {
                loadingState.classList.add("d-none");
                placeholderState.classList.remove("d-none");
                alert("An error occurred during classification. Check console for details.");
            }
        } finally {
            isRequestInProgress = false;
            if (pendingRequest) {
                pendingRequest = false;
                executeClassification(true);
            }
        }
    }
});

// Function to select sample image
function selectSample(sampleUrl) {
    const samplePathInput = document.getElementById("samplePath");
    const imageInput = document.getElementById("imageInput");
    const selectedFileName = document.getElementById("selectedFileName");
    const cachedFilenameInput = document.getElementById("cachedFilename");

    if (imageInput) imageInput.value = "";
    if (cachedFilenameInput) cachedFilenameInput.value = "";
    if (samplePathInput) samplePathInput.value = sampleUrl;
    if (selectedFileName) selectedFileName.textContent = `Sample: ${sampleUrl.split("/").pop()}`;

    // Highlight thumbnail
    document.querySelectorAll(".sample-thumb").forEach(el => {
        el.classList.remove("selected", "selected-ham");
        if (el.getAttribute("src") === sampleUrl) {
            if (sampleUrl.includes("spam")) {
                el.classList.add("selected");
            } else {
                el.classList.add("selected-ham");
            }
        }
    });

    // Auto-trigger submit for instant feedback
    const classifyForm = document.getElementById("classifyForm");
    if (classifyForm) {
        classifyForm.dispatchEvent(new Event("submit", { cancelable: true }));
    }
}
