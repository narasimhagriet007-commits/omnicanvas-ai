import os
import sys
import time
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch
import streamlit as st

# ML & AI imports
from diffusers import AutoPipelineForText2Image
from transformers import CLIPProcessor, CLIPModel
import chromadb
from chromadb.config import Settings

# =====================================================================
# 1. HARDWARE & INITIALIZATION MODULE
# =====================================================================
@st.cache_resource
def load_hardware_context():
    """Detect GPU hardware and configure optimal PyTorch precision."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype

DEVICE, DTYPE = load_hardware_context()

# =====================================================================
# 2. VISION JUDGE MODULE (CLIP Cosine Similarity Evaluator)
# =====================================================================
class VisionQualityJudge:
    """Uses OpenAI CLIP to evaluate semantic alignment between prompt and generated image."""
    def __init__(self, device: str):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def evaluate(self, image: Image.Image, text_prompt: str) -> float:
        """Returns an alignment score between 0.0 and 100.0%"""
        inputs = self.processor(text=[text_prompt], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_embeds = self.model.get_image_features(inputs['pixel_values'])
            text_embeds = self.model.get_text_features(inputs['input_ids'])

        # Normalize vectors
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        similarity = (image_embeds @ text_embeds.T).item()
        return round(max(0.0, similarity) * 100, 2)

# =====================================================================
# 3. HIGH-PERFORMANCE IMAGE GENERATION PIPELINE
# =====================================================================
class GenerativeImageEngine:
    """Engine responsible for loading SDXL Turbo / Stable Diffusion models."""
    def __init__(self, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=self.dtype, 
            variant="fp16" if device == "cuda" else None
        )
        self.pipe.to(self.device)

    def generate(self, prompt: str, num_inference_steps: int = 2) -> Image.Image:
        """Generates an image based on refined prompt string."""
        image = self.pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=0.0).images[0]
        return image

# =====================================================================
# 4. RAG VECTOR DATABASE ENGINE (Memory Subsystem)
# =====================================================================
class CampaignKnowledgeBase:
    """Vector Database storing winning aesthetic frameworks and prompt templates."""
    def __init__(self):
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name="creative_styles")
        self._seed_knowledge_base()

    def _seed_knowledge_base(self):
        """Seed vector DB with high-converting visual design principles."""
        if self.collection.count() == 0:
            styles = [
                "Cyberpunk futuristic neon lighting, highly detailed, 8k resolution, cinematic atmosphere, octane render",
                "Minimalist studio product photography, clean backdrop, soft ambient lighting, high commercial quality",
                "Surrealist digital art, vibrant colors, intricate details, trend on ArtStation, dynamic compositions",
                "Photorealistic editorial portrait, 85mm lens, natural daylight, soft bokeh, award-winning photography"
            ]
            metadatas = [{"style_type": "sci-fi"}, {"style_type": "product"}, {"style_type": "artistic"}, {"style_type": "portrait"}]
            ids = ["style_1", "style_2", "style_3", "style_4"]
            self.collection.add(documents=styles, metadatas=metadatas, ids=ids)

    def query_context(self, user_brief: str) -> str:
        """Retrieves best artistic prompt prefix based on user intent."""
        results = self.collection.query(query_texts=[user_brief], n_results=1)
        if results["documents"] and len(results["documents"][0]) > 0:
            return results["documents"][0][0]
        return "High quality, hyper-detailed, award winning aesthetics"

# =====================================================================
# 5. AUTONOMOUS AGENT ORCHESTRATOR (Self-Correction Loop)
# =====================================================================
class AutonomousVisualAgent:
    """Agent that handles generation, visual feedback evaluation, and prompt auto-refinement."""
    def __init__(self):
        self.device, self.dtype = DEVICE, DTYPE
        self.rag = CampaignKnowledgeBase()
        self.gen_engine = GenerativeImageEngine(self.device, self.dtype)
        self.evaluator = VisionQualityJudge(self.device)

    def refine_prompt(self, base_prompt: str, context: str, iteration: int) -> str:
        """Simulates an LLM-driven prompt enhancement routine."""
        modifiers = [
            "ultra-sharp focus, volumetric lighting, photorealistic textures",
            "trending on ArtStation, master masterpiece, trending color grade",
            "unreal engine 5 render, raytracing enabled, perfect composition"
        ]
        modifier = modifiers[min(iteration, len(modifiers) - 1)]
        return f"{base_prompt}, {context}, {modifier}"

    def execute_autonomous_loop(self, user_brief: str, target_score: float = 25.0, max_iterations: int = 3, log_container=None):
        logs = []
        best_image = None
        best_score = -1.0
        best_prompt = ""

        # Step 1: Vector RAG Retrieval
        context_style = self.rag.query_context(user_brief)
        logs.append(f"🧠 **[RAG Memory Engine]** Retrieved Style Context: *'{context_style}'*")
        if log_container:
            log_container.markdown("<br>".join(logs), unsafe_allow_html=True)

        current_prompt = f"{user_brief}, {context_style}"

        for i in range(max_iterations):
            logs.append(f"🔄 **[Iteration {i+1}/{max_iterations}]** Generating with prompt: *'{current_prompt}'*")
            if log_container:
                log_container.markdown("<br>".join(logs), unsafe_allow_html=True)

            # Step 2: Image Generation
            img = self.gen_engine.generate(current_prompt)

            # Step 3: Computer Vision Evaluation
            score = self.evaluator.evaluate(img, user_brief)
            logs.append(f"👁️ **[Vision Quality Judge]** CLIP Alignment Score: **{score}%** (Target: {target_score}%)")
            if log_container:
                log_container.markdown("<br>".join(logs), unsafe_allow_html=True)

            if score > best_score:
                best_score = score
                best_image = img
                best_prompt = current_prompt

            # Step 4: Decision & Autonomous Reflection
            if score >= target_score:
                logs.append(f"✅ **[Agent Decision]** Quality threshold satisfied! Finalizing output.")
                if log_container:
                    log_container.markdown("<br>".join(logs), unsafe_allow_html=True)
                break
            else:
                logs.append(f"⚠️ **[Agent Reflection]** Score insufficient. Triggering auto-refinement loop...")
                if log_container:
                    log_container.markdown("<br>".join(logs), unsafe_allow_html=True)
                current_prompt = self.refine_prompt(user_brief, context_style, i + 1)

        return best_image, best_score, best_prompt, logs

# Cache the agent instance to avoid reloading models on every button click
@st.cache_resource
def get_autonomous_agent():
    return AutonomousVisualAgent()

# =====================================================================
# 6. STREAMLIT ENTERPRISE DASHBOARD INTERFACE
# =====================================================================
def main():
    st.set_page_config(page_title="OmniCanvas AI Agent", page_icon="🎨", layout="wide")

    st.title("⚡ OmniCanvas AI: Autonomous Multi-Modal Visual Agent")
    st.caption("Grand-Prix Hackathon Architecture: RAG + SDXL + Real-time CLIP Feedback Loop")

    # Sidebar Controls
    st.sidebar.header("Agent Settings")
    quality_threshold = st.sidebar.slider("CLIP Target Quality Score (%)", min_value=15.0, max_value=40.0, value=24.0, step=0.5)
    max_loops = st.sidebar.slider("Max Self-Correction Loops", min_value=1, max_value=5, value=3)

    # Main Input Interface
    user_prompt = st.text_input("Enter your creative campaign brief:", value="A futuristic sleek electric sports car driving through a rainy neon Tokyo street")

    if st.button("🚀 Launch Autonomous Agent Loop", type="primary"):
        if not user_prompt.strip():
            st.warning("Please input a prompt.")
            return

        with st.spinner("Initializing Multi-Agent Engine..."):
            agent = get_autonomous_agent()

        col_logs, col_output = st.columns([1, 1])

        with col_logs:
            st.subheader("🖥️ Agent Execution & Reasoning Logs")
            log_container = st.empty()

        with col_output:
            st.subheader("🖼️ Final Evaluated Asset")
            image_container = st.empty()
            metric_container = st.empty()

        start_time = time.time()
        final_img, final_score, final_prompt, logs = agent.execute_autonomous_loop(
            user_brief=user_prompt,
            target_score=quality_threshold,
            max_iterations=max_loops,
            log_container=log_container
        )
        elapsed_time = round(time.time() - start_time, 2)

        image_container.image(final_img, use_container_width=True)
        metric_container.success(f"🎯 **Best Quality Score:** {final_score}% | ⏱️ **Total Execution Time:** {elapsed_time}s")
        st.info(f"**Final Prompt Used:** {final_prompt}")

if __name__ == "__main__":
    main()