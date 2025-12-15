# detailed_architecture_steps.md

| Step | Component | Input Shape | Output Shape | What happens? | Visual Representation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Input** | `(3, 64, 64)` | `(3, 64, 64)` | Raw RGB Image. | A square image icon. |
| **2** | **ResNet18 Backbone** | `(3, 64, 64)` | `(512, 2, 2)` | Extracts deep features. The spatial size shrinks (64->2), channels increase (3->512). | A large rectangular block labeled "CNN Feature Extractor". |
| **3** | **1x1 Convolution** | `(512, 2, 2)` | `(256, 2, 2)` | **Fusion Adapter**. Reduces channels from 512 to 256 to match the Transformer's expected size. | A small narrow block labeled "Projection (1x1 Conv)". |
| **4** | **Flatten** | `(256, 2, 2)` | `(4, 256)` | Reshapes the grid (2x2) into a sequence of 4 items. The Transformer needs a sequence, not a grid. | An arrow labeled "Flatten to Sequence". |
| **5** | **Transformer Encoder** | `(4, 256)` | `(4, 256)` | **Global Context**. The 4 patches "look at each other" using Self-Attention to understand the whole image structure. | A block labeled "Transformer (2 Layers)". |
| **6** | **Global Avg Pooling** | `(4, 256)` | `(256)` | Averages the 4 sequence items into a single vector representing the whole image. | A trapezoid or funnel shape labeled "Average Pool". |
| **7** | **Linear Classifier** | `(256)` | `(10)` | Maps the final vector to the 10 class probabilities (Dog, Cat, etc.). | A final block labeled "Classifier". |

---
