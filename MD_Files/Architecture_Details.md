# detailed_architecture_steps.md

Use this file to build your diagram and explain your Methodology in the report.

## 1. High-Level Data Flow (The "Story" of the Tensor)

This is exactly what happens to a single image as it passes through your model.

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

## 2. Step-by-Step Drawing Guide (For PowerPoint / Draw.io)

Follow these exact steps to create a professional diagram:

**Step 1: The Input**
*   Draw a **Square** on the far left.
*   Label it: **"Input Image (64x64)"**.
*   *Tip: Paste a small sample picture of a frog or plane here.*

**Step 2: The CNN (Local Features)**
*   Draw an arrow from Input to a **Large Blue Rectangle**.
*   Label it: **"ResNet-18 Backbone"**.
*   Inside or below it, write: *"Extracts Texture & Shapes"*.

**Step 3: The Bridge (Fusion)**
*   Draw an arrow from the CNN to a **Small Orange Rectangle**.
*   Label it: **"1x1 Conv"**.
*   Draw an arrow leaving this box labeled **"Flatten"**.
*   *Explanation*: This converts the 2D pictures into a sequence for the Transformer.

**Step 4: The Transformer (Global Context)**
*   Draw a **Purple Rectangle** (make it look distinct from the CNN).
*   Label it: **"Transformer Encoder"**.
*   Inside, add bullet points:
    *   *2 Layers*
    *   *Self-Attention*
    *   *Global Reasoning*

**Step 5: The Output**
*   Draw an arrow to a **Green Circle/Node**.
*   Label it: **"Pooling + Softmax"**.
*   Draw 10 small lines coming out of it.
*   Label them: **"Class Predictions"**.

---

## 3. Method Description for Report
*Copy-paste and tweak this for your "Methodology - Architecture" section:*

> "Our architecture fuses a Convolutional Neural Network (CNN) with a Transformer. We use a **ResNet-18** purely as a feature extractor to process the $64 \times 64$ input images. The ResNet outputs a feature map of shape $(512, 2, 2)$, capturing rich local spatial information.
>
> To interface with the Transformer, we apply a **$1 \times 1$ Convolution** to reduce the dimensionality to $256$, followed by flattening the spatial dimensions into a sequence of length $4$. This sequence is passed to a **2-layer Transformer Encoder**, which utilizes self-attention to model global relationships between the feature patches. Finally, we apply global average pooling and a linear classifier to produce the predictions for the $10$ CIFAR-10 classes."
