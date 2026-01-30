import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "spritegrid.pixelatedPreview",
    async setup() {
        const style = document.createElement("style");
        style.textContent = `
            img,
            canvas {
                image-rendering: pixelated;
                image-rendering: -moz-crisp-edges;
                image-rendering: crisp-edges;
            }
        `;
        document.head.appendChild(style);
    }
});
