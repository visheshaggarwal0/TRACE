# TRACE: Viva Defense Cheat-Sheet & Oral Exam Preparation

**Document ID:** `TRACE-DOC-004`  
**Version:** `1.0.0`  
**System Name:** **TRACE** (*Topological Road Augmentation and Completion Engine*)

---

## 1. The 60-Second Opening Pitch

> *"Good morning, Professors. Our project is **TRACE: Topological Road Augmentation and Completion Engine**.
>
> While satellite imagery is updated daily, keeping digital navigation maps up to date is a slow, manual bottleneck that lags reality by up to 18 months. Existing academic models treat road mapping as a **blank-slate problem**—trying to re-extract all roads from scratch—and evaluate solely on pixel-level overlap (IoU). This produces fragmented pixel masks that break under tree canopies and cannot be navigated by routing engines.
>
> **TRACE shifts the paradigm from blank-slate segmentation to prior-conditioned incremental graph completion.** We feed the neural network a satellite tile alongside an existing, incomplete OpenStreetMap vector layer as a spatial prior. The network is trained with connectivity-aware loss (**clDice**) to detect only the missing road residual. We then pass these predictions through a **deterministic tangent-snapping engine** that vectorizes the lines, collapses junction clusters, and snaps new roads into the master network.
>
> We evaluate our system on **APLS (Average Path Length Similarity)** to prove that the resulting graph is immediately traversable by routing engines like Dijkstra and A*."*

---

## 2. Top 15 Viva Questions & Bulletproof Model Answers

### Q1: *"What is the difference between a raster mask and a vector graph?"*
* **Answer:** *"A raster mask is a grid of pixels with color or probability values; it has no semantic understanding of connectivity, road width, or intersections. A vector graph is a mathematical network $G = (V, E)$ where vertices represent intersections and edges represent traversable road corridors with physical lengths in meters. Navigation engines like OSRM or Dijkstra cannot calculate routes on pixel grids; they fundamentally require vector graphs."*

### Q2: *"Why did you use D-LinkNet instead of standard U-Net?"*
* **Answer:** *"Standard U-Net relies on aggressive pooling downsampling, which destroys thin 2-to-3-pixel road boundaries at the bottleneck. D-LinkNet uses a central dilated convolution block with parallel dilation rates of 1, 2, 4, and 8. This expands the receptive field to capture long highways across the tile without reducing spatial resolution, while using skip connections to preserve fine edge details."*

### Q3: *"How does clDice work, and why is it differentiable during backpropagation?"*
* **Answer:** *"Classical skeletonization is discrete and non-differentiable. Soft-clDice makes skeletonization differentiable by formulating morphological erosion and dilation as continuous pooling layers: erosion is negative max-pooling of negative inputs (min-pooling), and dilation is standard max-pooling. clDice then computes the topological precision and sensitivity of the predicted soft skeleton versus the ground truth skeleton. A 2-pixel break in a road centerline snaps the skeleton, incurring a massive gradient penalty even when pixel IoU is over 99%."*

### Q4: *"Isn't using an existing road map already done by P2CNet in 2023?"*
* **Answer:** *"Yes, P2CNet in IEEE TGRS 2023 demonstrated that partial road maps help neural networks detect missing road pixels. However, P2CNet stopped at the **raster perception stage**—it only output a pixel image, evaluated strictly on pixel IoU, and left the newly predicted roads floating disconnected from the master map. Our novelty is taking this into a complete **topological routing pipeline**: we integrate clDice connectivity loss, implement an automated tangent-snapping engine, and evaluate on APLS navigability."*

### Q5: *"Why did you simulate an incomplete basemap by deleting 30% of roads?"*
* **Answer:** *"30% edge deletion is an empirically balanced benchmark for testing incremental map recovery. If you delete only 5%, the task is trivial; if you delete 80%, there is insufficient spatial prior structure left to guide the network. We delete whole connected linestring segments rather than random pixels to realistically simulate an unmapped housing layout or peripheral suburb."*

### Q6: *"What is APLS and how does it differ from pixel IoU?"*
* **Answer:** *"Pixel IoU measures the area overlap of road pixels; it is completely blind to connectivity. APLS (Average Path Length Similarity) measures true network navigability. It samples hundreds of random point pairs across the ground truth graph, computes the shortest Dijkstra driving path between them, and compares that distance to the shortest path on the predicted graph. If a road has a 2-pixel gap under a tree, Dijkstra cannot find a route or forces a 5-mile detour, driving the APLS score to zero."*

### Q7: *"How does your Snapping Engine work?"*
* **Answer:** *"The vectorizer extracts newly discovered road segments, but shadow occlusion at intersections often leaves a 5-to-15-meter air gap before touching the master highway. The Snapping Engine detects degree-1 boundary endpoints, computes their approach unit tangent vector, searches within a 30-meter radius for the nearest master highway edge, projects the point orthogonally onto the master line, splits the master edge, and inserts a valid junction node."*

### Q8: *"What happens if your model predicts a false positive road that doesn't exist?"*
* **Answer:** *"In our current pipeline, we filter false positives by applying a high confidence threshold ($p \ge 0.50$) and pruning isolated components shorter than 15 meters. In a production enterprise setting, you would cross-reference the candidate road against a second satellite tile acquired on a different date before automatically merging it into the master database."*

### Q9: *"How do you handle highway overpasses and bridges where roads cross without intersecting?"*
* **Answer:** *"In 2D planar skeletonization, crossovers appear as 4-way junctions. Our tangent-continuity filter inspects the angular heading of crossing lines: if two straight lines cross with zero turning curvature and approach angles near $90^\circ$, they are flagged as candidate grade-separated crossings rather than turnable intersections. In our future work, we plan to use shadow and height clues to assign vertical z-levels ($z=0$ vs. $z=1$)."*

### Q10: *"What coordinate systems do you use and why?"*
* **Answer:** *"We use UTM (Universal Transverse Mercator, e.g., EPSG:32611) because it is a conformal, projected coordinate system measured in **metric units (meters)**. Real-world GPS coordinates in WGS 84 (EPSG:4326) are angular (degrees), making physical distance and buffer calculations geometrically distorted."*

### Q11: *"What is degree-2 edge contraction?"*
* **Answer:** *"When you skeletonize an image, every pixel is initially treated as a node, producing over 100,000 nodes for a single tile. A node with degree 2 simply represents a point along a continuous straight road. Degree-2 contraction removes these intermediary nodes and merges their incident edges into a single polyline edge, accumulating their physical lengths. This reduces graph size by over 99% without losing geometric shape."*

### Q12: *"What is DBSCAN junction consolidation?"*
* **Answer:** *"At complex, wide intersections, skeletonization often generates a messy cluster of 3 to 5 closely spaced nodes forming artificial triangles and loops. We run DBSCAN with a 10-meter spatial radius to identify these clusters, collapse them into a single centroid node, and re-anchor all connected edges."*

### Q13: *"What is spur pruning?"*
* **Answer:** *"Noisy segmentation boundaries create tiny 1-to-5-pixel branches sticking out sideways from roads like whiskers. Spur pruning identifies dead-end nodes ($\text{Degree} = 1$) whose total edge length is under a threshold (e.g., 15 meters) and trims them off."*

### Q14: *"If the 4th channel has known roads, won't the model just copy them?"*
* **Answer:** *"No, because of our **residual target formulation**. The ground truth label provided to the network is $Y_{\text{target}} = Y_{\text{ground\_truth}} \setminus M_{\text{prior}}$. The model receives a loss penalty if it reproduces roads that are already in the 4th channel. It learns to use Channel 4 as an inhibitory guide: only predicting roads where Channel 4 is zero."*

### Q15: *"What is your main research takeaway?"*
* **Answer:** *"That adding an inductive spatial prior (existing GIS maps) reduces search complexity and false positives, while optimizing for connectivity (clDice) and performing topological snapping directly closes the gap between computer vision perception and real-world vehicle routing."*

---

## 3. Red Flags & Pitfalls to Avoid in Your Viva

* ❌ **Do NOT say:** *"We invented a brand new deep learning architecture from scratch."*  
  👉 **Say instead:** *"We designed a novel, prior-conditioned problem formulation and end-to-end topological snapping pipeline."*
* ❌ **Do NOT say:** *"Pixel IoU is the best metric to measure road quality."*  
  👉 **Say instead:** *"Pixel IoU measures area overlap but is blind to connectivity; APLS measures actual shortest-path driving reachability."*
* ❌ **Do NOT say:** *"Nobody has ever thought of using incomplete maps before."*  
  👉 **Say instead:** *"P2CNet proved the value of partial maps for pixel segmentation, but stopped at the raster stage; we completed the pipeline into traversable vector topology."*
