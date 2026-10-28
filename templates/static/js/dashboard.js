const sensorNames = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_20",
    "sensor_21"
];

const photoFieldAliases = {
    failure: {
        "machine-type": ["product type", "type"],
        "air-temp": ["air temperature k", "air temperature", "air temp"],
        "process-temp": ["process temperature k", "process temperature", "process temp"],
        "rotational-speed": ["rotational speed rpm", "rotational speed", "speed rpm"],
        torque: ["torque nm", "torque"],
        "tool-wear": ["tool wear min", "tool wear"]
    },
    rul: {
        op1: ["operating setting 1", "op setting 1"],
        op2: ["operating setting 2", "op setting 2"],
        ...Object.fromEntries(
            sensorNames.map(name => [
                name,
                [name.replace("_", " ")]
            ])
        )
    }
};

const sensorContainer = document.getElementById("sensor-inputs");

// Create an input for each sensor required by the trained RUL model.
sensorNames.forEach((name, index) => {
    const group = document.createElement("div");
    group.className = "form-group";

    const label = document.createElement("label");
    label.htmlFor = name;
    label.textContent = name.replace("_", " ").toUpperCase();

    const input = document.createElement("input");
    input.id = name;
    input.type = "number";
    input.step = "any";
    input.required = true;

    // Starting values are illustrative placeholders, not live readings.
    input.value = ["642", "1580", "1400", "14.6", "21.6",
                   "554", "2388", "9040", "47.5", "521",
                   "2388", "8130", "8.4", "0.03", "392",
                   "39", "23.4"][index];

    group.appendChild(label);
    group.appendChild(input);
    sensorContainer.appendChild(group);
});

let sensorChart = null;

function showResult(element, type, title, lines) {
    element.className = `result-box ${type}`;
    element.replaceChildren();

    const heading = document.createElement("h3");
    heading.textContent = title;
    element.appendChild(heading);

    lines.forEach(line => {
        const paragraph = document.createElement("p");
        paragraph.textContent = line;
        element.appendChild(paragraph);
    });
}

function getNumber(id) {
    const value = document.getElementById(id).value.trim();

    if (value === "") {
        throw new Error("Please fill in every required field.");
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        throw new Error("Please enter valid numeric values.");
    }

    return number;
}

async function postJSON(url, payload) {
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    });

    const result = await response.json();

    if (!response.ok) {
        throw new Error(
            result.error || "The server could not complete the prediction."
        );
    }

    return result;
}

function addHistoryCell(row, value) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.appendChild(cell);
}

function normalizeOcrLabel(value) {
    return value
        .toLowerCase()
        .replace(/_/g, " ")
        .replace(/[()[\]:=]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function extractPhotoValues(text, model) {
    const lines = text.split(/\r?\n/).map(normalizeOcrLabel);
    const aliases = photoFieldAliases[model];
    const extracted = {};

    Object.entries(aliases).forEach(([fieldId, fieldAliases]) => {
        for (let index = 0; index < lines.length; index += 1) {
            const line = lines[index];
            const alias = fieldAliases
                .map(normalizeOcrLabel)
                .sort((left, right) => right.length - left.length)
                .find(candidate =>
                    line === candidate || line.startsWith(`${candidate} `)
                );

            if (!alias) {
                continue;
            }

            const remainingText = line.slice(alias.length);
            const inlineValue = remainingText.match(
                /^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?|[lmh])\b/i
            );
            const nextLineValue = index + 1 < lines.length
                ? lines[index + 1].match(
                    /^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?|[lmh])\b/i
                )
                : null;
            const match = inlineValue || nextLineValue;

            if (!match) {
                continue;
            }

            const rawValue = match[1];
            if (fieldId === "machine-type") {
                const productType = rawValue.toUpperCase();
                if (["L", "M", "H"].includes(productType)) {
                    extracted[fieldId] = productType;
                }
            } else {
                const number = Number(rawValue);
                if (Number.isFinite(number)) {
                    extracted[fieldId] = number;
                }
            }
            break;
        }
    });

    return extracted;
}

function clearModelInputs(model) {
    Object.keys(photoFieldAliases[model]).forEach(fieldId => {
        const input = document.getElementById(fieldId);
        input.value = "";
    });
}

function applyPhotoValues(model, extracted) {
    Object.entries(extracted).forEach(([fieldId, value]) => {
        document.getElementById(fieldId).value = String(value);
    });
}

let sensorPhotoPreviewUrl = null;
document.getElementById("sensor-photo").addEventListener(
    "change",
    function () {
        const preview = document.getElementById("sensor-photo-preview");
        const file = this.files[0];
        if (sensorPhotoPreviewUrl) {
            URL.revokeObjectURL(sensorPhotoPreviewUrl);
            sensorPhotoPreviewUrl = null;
        }

        if (!file) {
            preview.classList.add("hidden");
            preview.removeAttribute("src");
            return;
        }

        sensorPhotoPreviewUrl = URL.createObjectURL(file);
        preview.src = sensorPhotoPreviewUrl;
        preview.classList.remove("hidden");
    }
);

document.getElementById("read-sensor-photo").addEventListener(
    "click",
    async function () {
        const fileInput = document.getElementById("sensor-photo");
        const status = document.getElementById("photo-ocr-status");
        const recognizedText = document.getElementById("photo-ocr-text");
        const model = document.getElementById("photo-model").value;
        const file = fileInput.files[0];

        if (!file) {
            status.textContent = "Choose a sensor display photo first.";
            return;
        }

        if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
            status.textContent = "Choose a PNG, JPEG, or WebP image.";
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            status.textContent = "Choose an image smaller than 10 MB.";
            return;
        }

        if (!window.Tesseract) {
            status.textContent =
                "The OCR library did not load. Check your internet connection and refresh the page.";
            return;
        }

        const button = this;
        button.disabled = true;
        button.textContent = "Reading...";
        status.textContent =
            "Reading the photo in this browser. Large images may take a little while...";
        recognizedText.textContent = "";

        let worker;
        try {
            worker = await window.Tesseract.createWorker("eng");
            const result = await worker.recognize(file);
            const text = result.data.text;
            recognizedText.textContent = text || "No text was recognized.";

            clearModelInputs(model);
            const extracted = extractPhotoValues(text, model);
            applyPhotoValues(model, extracted);

            const requiredCount = Object.keys(photoFieldAliases[model]).length;
            const extractedCount = Object.keys(extracted).length;
            const modelLabel = model === "failure" ? "AI4I" : "NASA RUL";
            status.textContent =
                `${modelLabel}: recognized ${extractedCount} of ${requiredCount} labelled fields. ` +
                "Review the matching form, correct OCR mistakes, and fill any blank fields before predicting.";
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            status.textContent = `Could not read the image: ${message}`;
        } finally {
            if (worker) {
                await worker.terminate();
            }
            button.disabled = false;
            button.textContent = "Read Photo Values";
        }
    }
);

async function loadPredictionHistory() {
    const message = document.getElementById("history-message");
    const rowsContainer = document.getElementById("history-rows");
    const typeFilter = document.getElementById("history-type-filter");
    const parameters = new URLSearchParams({ limit: "50" });

    if (typeFilter.value) {
        parameters.set("prediction_type", typeFilter.value);
    }

    message.className = "empty-message";
    message.textContent = "Loading prediction history...";

    try {
        const response = await fetch(`/api/predictions/history?${parameters}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Could not load prediction history."
            );
        }

        rowsContainer.replaceChildren();

        if (data.predictions.length === 0) {
            message.textContent = "No predictions saved yet.";
            return;
        }

        data.predictions.forEach(record => {
            const row = document.createElement("tr");
            const result = record.predicted_result || {};
            const resultText = record.prediction_type === "failure"
                ? result.status
                : `${result.predicted_rul_cycles} cycles`;

            addHistoryCell(row, String(record.prediction_id));
            addHistoryCell(row, record.machine_id || "Unassigned");
            addHistoryCell(
                row,
                record.prediction_type === "failure" ? "Failure" : "RUL"
            );
            addHistoryCell(row, resultText || "—");
            addHistoryCell(
                row,
                record.failure_probability === null
                    ? "—"
                    : `${record.failure_probability}%`
            );
            addHistoryCell(
                row,
                record.predicted_rul === null
                    ? "—"
                    : `${record.predicted_rul} cycles`
            );
            addHistoryCell(row, record.risk_level || "Not assessed");
            addHistoryCell(row, record.model_name);
            addHistoryCell(row, record.created_at);
            rowsContainer.appendChild(row);
        });

        message.textContent = `Showing ${data.count} saved prediction(s).`;
    } catch (error) {
        message.className = "empty-message history-error";
        message.textContent = error.message;
    }
}

// Check whether both model artifacts load in the Flask application.
async function checkBackend() {
    const rulStatus = document.getElementById("rul-status");
    const failureStatus = document.getElementById("failure-status");
    const connectionStatus = document.getElementById("backend-connection-status");

    try {
        const response = await fetch("/api/health");
        const data = await response.json();

        if (!response.ok || data.status !== "success") {
            throw new Error("Backend health check failed.");
        }

        rulStatus.textContent = data.rul_model_loaded ? "Ready" : "Unavailable";
        failureStatus.textContent = data.failure_model_loaded
            ? "Ready"
            : "Unavailable";
        connectionStatus.textContent = data.history_database_connected
            ? "Backend and history DB connected"
            : "Backend online; history DB offline";
        await loadPredictionHistory();

    } catch (error) {
        rulStatus.textContent = "Offline";
        failureStatus.textContent = "Offline";
        connectionStatus.textContent = "Backend offline";
        console.error("Backend health check:", error);
    }
}

// Failure prediction
document.getElementById("failure-form").addEventListener(
    "submit",
    async function (event) {
        event.preventDefault();

        const resultBox = document.getElementById("failure-result");
        const button = this.querySelector("button[type='submit']");

        button.disabled = true;
        button.textContent = "Analyzing...";

        try {
            const payload = {
                "Type": document.getElementById("machine-type").value,
                "Air temperature [K]": getNumber("air-temp"),
                "Process temperature [K]": getNumber("process-temp"),
                "Rotational speed [rpm]": getNumber("rotational-speed"),
                "Torque [Nm]": getNumber("torque"),
                "Tool wear [min]": getNumber("tool-wear")
            };

            const result = await postJSON("/api/predict/failure", payload);

            const isFailure = result.predicted_failure === 1;
            const resultLines = [
                `Predicted class: ${isFailure ? "Failure" : "No failure"}`,
                `Estimated failure probability: ${result.failure_probability_percent}%`,
                `Model: ${result.model}`,
                "This is a model estimate, not a guarantee of actual machine condition."
            ];
            if (result.history_warning) {
                resultLines.push(result.history_warning);
            }

            showResult(
                resultBox,
                isFailure || !result.history_saved ? "warning" : "success",
                isFailure ? "Failure Risk Detected" : "No Failure Predicted",
                resultLines
            );

            document.getElementById("risk-summary").textContent =
                `${result.failure_probability_percent}%`;
            await loadPredictionHistory();

        } catch (error) {
            showResult(resultBox, "error", "Prediction Error", [error.message]);
        } finally {
            button.disabled = false;
            button.textContent = "Analyze Failure Risk →";
        }
    }
);

// Update the sensor chart using readings submitted to the RUL form.
function updateSensorChart(payload) {
    const labels = sensorNames;
    const values = sensorNames.map(name => payload[name]);

    const canvas = document.getElementById("sensor-chart");
    const message = document.getElementById("chart-message");

    message.textContent =
        "Showing the sensor values submitted for the latest RUL prediction.";

    if (sensorChart) {
        sensorChart.destroy();
    }

    sensorChart = new Chart(canvas, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Submitted sensor readings",
                data: values,
                borderWidth: 2,
                tension: 0.25,
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// NASA RUL prediction
document.getElementById("rul-form").addEventListener(
    "submit",
    async function (event) {
        event.preventDefault();

        const resultBox = document.getElementById("rul-result");
        const button = this.querySelector("button[type='submit']");

        button.disabled = true;
        button.textContent = "Estimating...";

        try {
            const payload = {
                op_setting_1: getNumber("op1"),
                op_setting_2: getNumber("op2")
            };

            sensorNames.forEach(name => {
                payload[name] = getNumber(name);
            });

            const result = await postJSON("/api/predict/rul", payload);
            const resultLines = [
                `Estimated remaining useful life: ${result.predicted_rul_cycles} cycles`,
                `Model: ${result.model}`,
                "This is a regression estimate based on the submitted sensor values.",
                "It is not a guarantee of the machine's actual remaining life."
            ];
            if (result.history_warning) {
                resultLines.push(result.history_warning);
            }

            showResult(
                resultBox,
                result.history_saved ? "success" : "warning",
                "RUL Estimate Generated",
                resultLines
            );

            document.getElementById("rul-summary").textContent =
                `${result.predicted_rul_cycles} cycles`;

            updateSensorChart(payload);
            await loadPredictionHistory();

        } catch (error) {
            showResult(resultBox, "error", "Prediction Error", [error.message]);
        } finally {
            button.disabled = false;
            button.textContent = "Estimate Remaining Useful Life →";
        }
    }
);

document.getElementById("refresh-history").addEventListener(
    "click",
    loadPredictionHistory
);
document.getElementById("history-type-filter").addEventListener(
    "change",
    loadPredictionHistory
);

checkBackend();