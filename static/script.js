const uploadBox = document.getElementById("uploadBox");
const photoInput = document.getElementById("photoInput");
const previewImg = document.getElementById("previewImg");
const uploadText = document.getElementById("uploadText");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultCard = document.getElementById("resultCard");
const resultContent = document.getElementById("resultContent");
const loadingSpinner = document.getElementById("loadingSpinner");

let selectedFile = null;

uploadBox.addEventListener("click", () => photoInput.click());

photoInput.addEventListener("change", (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        const reader = new FileReader();
        reader.onload = (evt) => {
            previewImg.src = evt.target.result;
            previewImg.style.display = "block";
            uploadText.style.display = "none";
        };
        reader.readAsDataURL(selectedFile);
    }
});

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        alert("Pehle ek photo upload karo!");
        return;
    }

    const clothingType = document.getElementById("clothingType").value || "shirt";
    const size = document.getElementById("size").value;
    const occasion = document.getElementById("occasion").value;

    const formData = new FormData();
    formData.append("photo", selectedFile);
    formData.append("clothing_type", clothingType);
    formData.append("size", size);
    formData.append("occasion", occasion);

    resultCard.style.display = "block";
    loadingSpinner.style.display = "block";
    resultContent.innerHTML = "";

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        loadingSpinner.style.display = "none";

        if (data.error) {
            resultContent.innerHTML = `<p style="color:#ff6b6b;">${data.error}</p>`;
            return;
        }

        if (!data.products || data.products.length === 0) {
            resultContent.innerHTML = `<p style="color:#ff6b6b;">Could not fetch suggestions. Try again.</p>`;
            return;
        }

        let accessoryHtml = "";
        if (data.accessory_guidance) {
            const acc = data.accessory_guidance;
            accessoryHtml = `
                <div style="margin-bottom:20px; padding:14px; background:#1a1a1a; border-radius:10px;">
                    <p style="font-size:13px; color:#999; margin-bottom:8px;"><strong>Accessory Guidance</strong></p>
                    ${acc.watch ? `<p style="font-size:13px; margin-bottom:4px;">⌚ Watch: ${acc.watch}</p>` : ""}
                    ${acc.shoes ? `<p style="font-size:13px; margin-bottom:4px;">👞 Shoes: ${acc.shoes}</p>` : ""}
                    ${acc.belt ? `<p style="font-size:13px;">👔 Belt: ${acc.belt}</p>` : ""}
                </div>
            `;
        }

        let cardsHtml = data.products.map(p => `
            <div class="product-card">
                <div class="product-info">
                    <h3>${p.name || "Product"}</h3>
                    <p class="brand">${p.brand || ""} ${p.color ? "· " + p.color : ""}</p>
                    <p class="price">${p.price || ""}</p>
                </div>
                <a href="${p.link || '#'}" target="_blank" class="buy-btn">Buy Now →</a>
            </div>
        `).join("");

        resultContent.innerHTML = `
            <p style="margin-bottom:16px; color:#ccc;"><strong>Detected Skin Tone:</strong> ${data.skin_tone.toUpperCase()} (${data.confidence}% confidence)</p>
            ${accessoryHtml}
            <div class="products-grid">${cardsHtml}</div>
        `;
    } catch (err) {
        loadingSpinner.style.display = "none";
        resultContent.innerHTML = `<p style="color:#ff6b6b;">Something went wrong. Try again.</p>`;
    }
});