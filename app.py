import base64
import hashlib
import io
import time
import numpy as np
import scipy.signal
import soundfile as sf
import streamlit as st

from cart import OrderCart
from menu_api import MenuAPIClient
from agent import ChefAgent
from vad import SileroVADDetector
from stt import WhisperSTT
from tts import TextToSpeech

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Page Setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Girfalco — Chef Voice Agent",
    page_icon="👨‍🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Simple, High-Contrast Kiosk Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark Kiosk Theme */
    .stApp {
        background-color: #0b0f19;
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    footer, #MainMenu {
        visibility: hidden !important;
    }

    /* Sidebar High Contrast Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d121f !important;
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] .stCaption p {
        color: #94a3b8 !important;
    }

    /* Button Styling - High Contrast & Crisp */
    button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4) !important;
    }
    button[kind="primary"] p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    button[kind="secondary"] {
        background: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"] p {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"]:hover {
        background: #2563eb !important;
        border-color: #60a5fa !important;
    }
    button[kind="secondary"]:hover p {
        color: #ffffff !important;
    }

    /* Hide visible audio player pop-up bars so the response voice plays seamlessly in realtime */
    div[data-testid="stAudio"], audio {
        display: none !important;
        height: 0px !important;
        visibility: hidden !important;
    }

    /* Premium Kiosk Styling for Browser Audio Input Widget */
    div[data-testid="stAudioInput"] {
        background: #111827 !important;
        border: 2px solid #10b981 !important;
        border-radius: 18px !important;
        padding: 14px 20px !important;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.25) !important;
        margin-top: 10px !important;
        margin-bottom: 16px !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stAudioInput"]:hover {
        border-color: #34d399 !important;
        box-shadow: 0 6px 25px rgba(52, 211, 153, 0.35) !important;
    }
    div[data-testid="stAudioInput"] label p {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #34d399 !important;
        letter-spacing: 0.3px !important;
    }

    /* Animated Voice Agent Movement Icon */
    .voice-stage {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 32px 20px;
        background: radial-gradient(circle at center, rgba(16, 185, 129, 0.12) 0%, rgba(11, 15, 25, 0) 70%);
        border: 1px solid #1f2937;
        border-radius: 24px;
        margin-bottom: 20px;
        position: relative;
    }

    .orb-wrapper {
        position: relative;
        width: 140px;
        height: 140px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .pulse-ring {
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        border: 2px solid rgba(52, 211, 153, 0.4);
        animation: radarPulse 2.8s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }

    .pulse-ring:nth-child(2) {
        animation-delay: 1s;
    }

    .agent-orb {
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #34d399 0%, #059669 50%, #064e3b 95%);
        box-shadow: 0 0 50px rgba(16, 185, 129, 0.6), inset 0 0 25px rgba(255, 255, 255, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 46px;
        color: #ffffff;
        z-index: 2;
        animation: orbFloat 3s ease-in-out infinite, orbGlow 2.2s ease-in-out infinite alternate;
        transition: all 0.3s ease;
    }

    .agent-orb.listening {
        background: radial-gradient(circle at 35% 35%, #fbbf24 0%, #d97706 50%, #78350f 95%) !important;
        box-shadow: 0 0 70px rgba(245, 158, 11, 0.8), inset 0 0 30px rgba(255, 255, 255, 0.7) !important;
        animation: orbActive 0.8s ease-in-out infinite alternate !important;
    }

    .agent-orb.user-speaking {
        background: radial-gradient(circle at 35% 35%, #fb7185 0%, #e11d48 50%, #881337 95%) !important;
        box-shadow: 0 0 85px rgba(244, 63, 94, 0.95), inset 0 0 35px rgba(255, 255, 255, 0.8) !important;
        animation: orbActive 0.45s ease-in-out infinite alternate !important;
    }

    .agent-orb.thinking {
        background: radial-gradient(circle at 35% 35%, #c084fc 0%, #7c3aed 50%, #4c1d95 95%) !important;
        box-shadow: 0 0 65px rgba(168, 85, 247, 0.85), inset 0 0 25px rgba(255, 255, 255, 0.7) !important;
        animation: orbFloat 1.2s ease-in-out infinite !important;
    }

    .agent-orb.understanding {
        background: radial-gradient(circle at 35% 35%, #c084fc 0%, #9333ea 50%, #581c87 95%) !important;
        box-shadow: 0 0 75px rgba(168, 85, 247, 0.9), inset 0 0 30px rgba(255, 255, 255, 0.75) !important;
        animation: orbActive 0.6s ease-in-out infinite alternate !important;
    }

    .agent-orb.checking {
        background: radial-gradient(circle at 35% 35%, #fbbf24 0%, #d97706 50%, #78350f 95%) !important;
        box-shadow: 0 0 75px rgba(245, 158, 11, 0.9), inset 0 0 30px rgba(255, 255, 255, 0.75) !important;
        animation: orbFloat 1.2s ease-in-out infinite alternate !important;
    }

    .agent-orb.speaking {
        background: radial-gradient(circle at 35% 35%, #38bdf8 0%, #0284c7 50%, #0c4a6e 95%) !important;
        box-shadow: 0 0 75px rgba(56, 189, 248, 0.9), inset 0 0 30px rgba(255, 255, 255, 0.7) !important;
        animation: orbActive 0.7s ease-in-out infinite alternate !important;
    }

    @keyframes radarPulse {
        0% { transform: scale(0.9); opacity: 0.8; }
        100% { transform: scale(1.7); opacity: 0; }
    }

    @keyframes orbFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }

    @keyframes orbGlow {
        0% { filter: drop-shadow(0 0 25px rgba(16, 185, 129, 0.4)); }
        100% { filter: drop-shadow(0 0 45px rgba(16, 185, 129, 0.8)); }
    }

    @keyframes orbActive {
        0% { transform: scale(0.96); }
        100% { transform: scale(1.08); }
    }

    .soundwave-bars {
        display: flex;
        align-items: center;
        gap: 6px;
        height: 28px;
        margin-top: 16px;
    }

    .soundwave-bar {
        width: 5px;
        height: 8px;
        background: #34d399;
        border-radius: 4px;
        animation: wavePulse 1.2s ease-in-out infinite;
        transition: background 0.3s ease;
    }

    .soundwave-bar:nth-child(2) { animation-delay: 0.15s; height: 16px; }
    .soundwave-bar:nth-child(3) { animation-delay: 0.3s; height: 26px; }
    .soundwave-bar:nth-child(4) { animation-delay: 0.45s; height: 18px; }
    .soundwave-bar:nth-child(5) { animation-delay: 0.2s; height: 10px; }

    .soundwave-bars.listening .soundwave-bar {
        background: #fbbf24 !important;
        animation: wavePulse 0.9s ease-in-out infinite !important;
    }

    .soundwave-bars.user-speaking .soundwave-bar {
        background: #f43f5e !important;
        animation: userWave 0.35s ease-in-out infinite alternate !important;
    }

    .soundwave-bars.understanding .soundwave-bar {
        background: #c084fc !important;
        animation: wavePulse 0.7s ease-in-out infinite !important;
    }

    .soundwave-bars.checking .soundwave-bar {
        background: #fbbf24 !important;
        animation: wavePulse 0.85s ease-in-out infinite !important;
    }

    .soundwave-bars.speaking .soundwave-bar {
        background: #38bdf8 !important;
        animation: wavePulse 0.65s ease-in-out infinite !important;
    }

    @keyframes wavePulse {
        0%, 100% { transform: scaleY(0.4); opacity: 0.6; }
        50% { transform: scaleY(1.4); opacity: 1; }
    }

    @keyframes userWave {
        0% { transform: scaleY(0.4); opacity: 0.7; }
        100% { transform: scaleY(2.2); opacity: 1; }
    }

    .agent-status-label {
        margin-top: 12px;
        font-size: 19px;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: color 0.3s ease;
    }

    /* Step Activity Pipeline Pills */
    .agent-step-pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin-top: 14px;
        flex-wrap: wrap;
    }

    .step-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        background: rgba(30, 41, 59, 0.6);
        color: #64748b;
        border: 1px solid rgba(51, 65, 85, 0.6);
        transition: all 0.3s ease;
    }

    .step-pill.active-understanding {
        background: rgba(168, 85, 247, 0.22) !important;
        color: #e9d5ff !important;
        border-color: #a855f7 !important;
        box-shadow: 0 0 16px rgba(168, 85, 247, 0.45) !important;
        transform: scale(1.06);
    }

    .step-pill.active-checking {
        background: rgba(245, 158, 11, 0.22) !important;
        color: #fef08a !important;
        border-color: #f59e0b !important;
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.45) !important;
        transform: scale(1.06);
    }

    .step-pill.active-speaking {
        background: rgba(56, 189, 248, 0.22) !important;
        color: #bae6fd !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.55) !important;
        transform: scale(1.06);
        animation: pulseSpeakingStep 1.1s ease-in-out infinite alternate;
    }

    @keyframes pulseSpeakingStep {
        0% { box-shadow: 0 0 8px rgba(56, 189, 248, 0.3); }
        100% { box-shadow: 0 0 22px rgba(56, 189, 248, 0.85); }
    }

    .step-arrow {
        color: #475569;
        font-size: 12px;
        font-weight: 800;
    }

    /* High Visibility Realtime Subtitles */
    .sub-card {
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    }

    .sub-card.customer {
        background: #0f172a;
        border: 2px solid #0284c7;
    }

    .sub-card.chef {
        background: #061a14;
        border: 2px solid #059669;
    }

    .sub-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .sub-badge-cust {
        background: rgba(56, 189, 248, 0.25);
        color: #38bdf8;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1px;
        padding: 3px 12px;
        border-radius: 20px;
    }

    .sub-badge-chef {
        background: rgba(52, 211, 153, 0.25);
        color: #34d399;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1px;
        padding: 3px 12px;
        border-radius: 20px;
    }

    .sub-time {
        font-size: 13px;
        color: #94a3b8;
    }

    .sub-text {
        font-size: 21px;
        font-weight: 600;
        line-height: 1.5;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session State Setup (Cart persists across page refreshes!)
# ─────────────────────────────────────────────────────────────────────────────
if "cart" not in st.session_state:
    st.session_state.cart = OrderCart()

if "menu_client" not in st.session_state:
    st.session_state.menu_client = MenuAPIClient(force_mock=False)

if "agent" not in st.session_state:
    st.session_state.agent = ChefAgent(cart=st.session_state.cart, menu_client=st.session_state.menu_client)

if "tts" not in st.session_state:
    st.session_state.tts = TextToSpeech()

if "vad" not in st.session_state:
    st.session_state.vad = SileroVADDetector()

if "stt" not in st.session_state:
    st.session_state.stt = WhisperSTT()

if "customer_subtitle" not in st.session_state:
    st.session_state.customer_subtitle = None

if "chef_subtitle" not in st.session_state:
    st.session_state.chef_subtitle = st.session_state.agent.get_greeting()
    st.session_state.chef_time = time.strftime("%H:%M:%S")

if "agent_status" not in st.session_state:
    st.session_state.agent_status = "Chef is Ready"

if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "idle"  # idle, listening, speaking

if "last_order_num" not in st.session_state:
    st.session_state.last_order_num = None

if "placed_order_info" not in st.session_state:
    st.session_state.placed_order_info = None

if "show_order_modal" not in st.session_state:
    st.session_state.show_order_modal = False

if "mic_key_idx" not in st.session_state:
    st.session_state.mic_key_idx = 0

if "_last_processed_audio_hash" not in st.session_state:
    st.session_state._last_processed_audio_hash = None

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Quick Menu Explorer (Manual Addition Kept & Fixed)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Quick Menu Explorer")
    st.caption("Browse categories and tap '+ Add' to manually add items without speaking.")
    st.divider()
    
    menu_res = st.session_state.menu_client.get_products()
    if menu_res.get("success"):
        prods = menu_res.get("products", [])
        cats = sorted(list(set(p.get("category", "General") for p in prods)))
        selected_cat = st.selectbox("Select Category", ["ALL"] + cats)
        
        filtered = prods if selected_cat == "ALL" else [p for p in prods if p.get("category") == selected_cat]
        
        for p in filtered:
            col_info, col_btn = st.columns([3, 1.2])
            with col_info:
                st.markdown(f"**{p['name']}**")
                st.caption(f"{p['price']:.3f} BHD")
            with col_btn:
                if st.button("+ Add", key=f"quick_add_{p['productId']}", use_container_width=True):
                    st.session_state.cart.add_item(
                        product_id=p["productId"],
                        name=p["name"],
                        quantity=1,
                        price=p["price"]
                    )
                    st.session_state.chef_subtitle = f"Added {p['name']} ({p['price']:.3f} BHD) to your order tray."
                    st.session_state.chef_time = time.strftime("%H:%M:%S")
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Order Placed Modal Dialog (Popup showing items, total amount, and order number)
# ─────────────────────────────────────────────────────────────────────────────
@st.dialog("🎉 Order Placed Successfully!")
def order_placed_dialog(order_info):
    st.balloons()
    st.markdown(f"""
    <div style="text-align: center; padding: 6px 0 14px 0;">
        <span style="font-size: 13px; font-weight: 800; color: #10b981; letter-spacing: 1.5px; text-transform: uppercase;">Confirmed Kitchen Ticket</span>
        <h1 style="font-size: 44px; font-weight: 900; color: #ffffff; letter-spacing: 2px; margin: 4px 0;">#{order_info.get('order_number')}</h1>
        <span style="font-size: 13px; color: #94a3b8;">Placed at {order_info.get('time', '')} • Sent directly to kitchen</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h4 style='color: #f8fafc; margin-bottom: 8px;'>📋 Items Placed</h4>", unsafe_allow_html=True)
    items = order_info.get("items", [])
    for it in items:
        c_name, c_price = st.columns([3, 1.2])
        with c_name:
            st.markdown(f"**{it.get('quantity', 1)}x {it.get('name')}**")
            for a in it.get("addons", []):
                st.markdown(f"<div style='font-size: 12px; color: #38bdf8; margin-left: 8px;'>+ {a.get('name')} (+{float(a.get('price', 0.5)):.3f} BHD)</div>", unsafe_allow_html=True)
            if it.get("notes") and not it.get("addons"):
                st.caption(f"Note: {it.get('notes')}")
        with c_price:
            price_val = float(it.get('total_price', it.get('price', 0.0)))
            st.markdown(f"<div style='text-align: right; font-weight: 700; color: #34d399;'>{price_val:.3f} BHD</div>", unsafe_allow_html=True)
        st.divider()

    total_amount = float(order_info.get("total", 0.0))
    st.markdown(f"""
    <div style="background: #0f172a; border: 1px solid #1f2937; border-radius: 12px; padding: 14px 18px; display: flex; justify-content: space-between; align-items: center; margin-top: 8px; margin-bottom: 20px;">
        <div>
            <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Status</span>
            <div style="font-size: 15px; font-weight: 700; color: #34d399;">✅ Confirmed & Sent to Counter</div>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Total Amount Paid</span>
            <div style="font-size: 26px; font-weight: 900; color: #34d399;">{total_amount:.3f} BHD</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; color: #cbd5e1; font-size: 14px;'>Please proceed to the counter to collect your order. Thank you for visiting!</p>", unsafe_allow_html=True)

    if st.button("Start New Order", type="primary", use_container_width=True):
        st.session_state.show_order_modal = False
        st.rerun()

# Trigger dialog if show_order_modal is True
if st.session_state.get("show_order_modal") and st.session_state.get("placed_order_info"):
    order_placed_dialog(st.session_state.placed_order_info)

# ─────────────────────────────────────────────────────────────────────────────
# Main Layout: 2 Columns (Left = Voice Agent & Subtitles, Right = Cart Tray)
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.5, 1.1], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# Left: Agent Speaking UI + Real-time Subtitles
# ─────────────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown("<h2 style='margin-bottom: 2px; color: #ffffff;'>👨‍🍳 Chef Voice Agent</h2>", unsafe_allow_html=True)
    st.caption("Speak to order food & drinks, ask questions, or customize items.")

    # Dynamic Voice Stage Renderer (Interactive Realtime Visualizer)
    def get_voice_stage_html(mode="idle", status_text=None):
        if mode == "listening":
            orb_class = "agent-orb listening"
            orb_icon = "🎙️"
            status_color = "#fbbf24"
            wave_class = "soundwave-bars listening"
            default_label = "Listening for your voice... Speak naturally"
            active_step = "understanding"
        elif mode == "user_speaking":
            orb_class = "agent-orb user-speaking"
            orb_icon = "🗣️"
            status_color = "#f43f5e"
            wave_class = "soundwave-bars user-speaking"
            default_label = "Hearing your voice... (Processing speech)"
            active_step = "understanding"
        elif mode == "understanding":
            orb_class = "agent-orb understanding"
            orb_icon = "🧠"
            status_color = "#c084fc"
            wave_class = "soundwave-bars understanding"
            default_label = "Agent is understanding what you said..."
            active_step = "understanding"
        elif mode == "checking":
            orb_class = "agent-orb checking"
            orb_icon = "📋"
            status_color = "#fbbf24"
            wave_class = "soundwave-bars checking"
            default_label = "Agent is checking the menu & orders..."
            active_step = "checking"
        elif mode == "speaking":
            orb_class = "agent-orb speaking"
            orb_icon = "🔊"
            status_color = "#38bdf8"
            wave_class = "soundwave-bars speaking"
            default_label = "Chef is speaking aloud..."
            active_step = "speaking"
        else:  # idle
            orb_class = "agent-orb"
            orb_icon = "👨‍🍳"
            status_color = "#34d399"
            wave_class = "soundwave-bars"
            default_label = "Chef is Ready — Tap microphone below to speak"
            active_step = "idle"

        label = status_text or default_label

        p_und = "active-understanding" if active_step == "understanding" else ""
        p_chk = "active-checking" if active_step == "checking" else ""
        p_spk = "active-speaking" if active_step == "speaking" else ""

        return f"""
        <div class="voice-stage">
            <div class="orb-wrapper">
                <div class="pulse-ring"></div>
                <div class="pulse-ring"></div>
                <div class="{orb_class}">
                    {orb_icon}
                </div>
            </div>
            <div class="{wave_class}">
                <div class="soundwave-bar"></div>
                <div class="soundwave-bar"></div>
                <div class="soundwave-bar"></div>
                <div class="soundwave-bar"></div>
                <div class="soundwave-bar"></div>
            </div>
            <div class="agent-status-label" style="color: {status_color};">
                ● {label}
            </div>
            <div class="agent-step-pipeline">
                <span class="step-pill {p_und}">🧠 1. Understanding</span>
                <span class="step-arrow">→</span>
                <span class="step-pill {p_chk}">📋 2. Checking Menu & Orders</span>
                <span class="step-arrow">→</span>
                <span class="step-pill {p_spk}">🔊 3. Chef Speaking</span>
            </div>
        </div>
        """

    voice_stage_slot = st.empty()
    voice_stage_slot.markdown(
        get_voice_stage_html(st.session_state.agent_mode, st.session_state.agent_status),
        unsafe_allow_html=True
    )

    def set_agent_ui_state(mode: str, label: str):
        st.session_state.agent_mode = mode
        st.session_state.agent_status = label
        voice_stage_slot.markdown(get_voice_stage_html(mode, label), unsafe_allow_html=True)

    # Seamless Real-time Voice Audio Playback in Browser (No visible audio player bar or pop-up)
    if getattr(st.session_state, "browser_speech_wav", None):
        audio_b64 = base64.b64encode(st.session_state.browser_speech_wav).decode()
        st.markdown(
            f"""
            <audio id="chef-voice-player" autoplay style="display:none;">
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>
            <script>
            (function() {{
                var audio = document.getElementById('chef-voice-player');
                if (audio) {{
                    audio.onended = function() {{
                        // Smoothly transition orb back to ready state when chef finishes speaking
                        var orb = document.querySelector('.agent-orb');
                        if (orb) {{
                            orb.className = 'agent-orb';
                            orb.innerHTML = '👨‍🍳';
                        }}
                        var lbl = document.querySelector('.agent-status-label');
                        if (lbl) {{
                            lbl.style.color = '#34d399';
                            lbl.innerHTML = '● Chef is Ready — Tap microphone below to speak';
                        }}
                        var waves = document.querySelector('.soundwave-bars');
                        if (waves) {{
                            waves.className = 'soundwave-bars';
                        }}
                        var pills = document.querySelectorAll('.step-pill');
                        pills.forEach(function(p) {{
                            p.classList.remove('active-speaking', 'active-understanding', 'active-checking');
                        }});
                    }};
                }}
            }})();
            </script>
            """,
            unsafe_allow_html=True
        )
        st.session_state.browser_speech_wav = None

    # Helper to process voice turns (prevents duplicate speech & infinite loops)
    def handle_voice_audio(audio_data, source="browser"):
        # Step 1: Agent is Understanding
        set_agent_ui_state("understanding", "Agent is understanding what you said...")
        status_slot = st.empty()
        status_slot.info("🧠 Transcribing your voice with Whisper...")
        transcript = st.session_state.stt.transcribe(audio_data)

        if transcript:
            st.session_state.customer_subtitle = transcript
            st.session_state.customer_time = time.strftime("%H:%M:%S")

            # Step 2: Agent is Checking Menu & Orders
            set_agent_ui_state("checking", "Agent is checking the menu & orders...")
            status_slot.warning("📋 Checking menu catalog, prices, and kitchen order tray...")
            reply = st.session_state.agent.process_turn(transcript)

            st.session_state.chef_subtitle = reply
            st.session_state.chef_time = time.strftime("%H:%M:%S")

            # Check if order was submitted during voice interaction
            if getattr(st.session_state.cart, "last_placed_order", None) and st.session_state.cart.is_submitted:
                st.session_state.placed_order_info = st.session_state.cart.last_placed_order
                st.session_state.last_order_num = st.session_state.placed_order_info.get("order_number")
                st.session_state.show_order_modal = True
                st.session_state.cart.is_submitted = False

            # Step 3: Chef is Speaking Aloud
            set_agent_ui_state("speaking", "Chef is speaking aloud...")

            # Synthesize audio exclusively for browser playback
            try:
                synth = st.session_state.tts.synthesize_audio(reply)
                if synth is not None:
                    audio_out, sr_out = synth
                    wav_buf = io.BytesIO()
                    sf.write(wav_buf, audio_out, sr_out, format="WAV")
                    st.session_state.browser_speech_wav = wav_buf.getvalue()
                else:
                    st.session_state.tts.speak(reply)
            except Exception:
                st.session_state.tts.speak(reply)

            # Keep in speaking state on rerun while browser plays audio
            st.session_state.agent_mode = "speaking"
            st.session_state.agent_status = "Chef speaking aloud..."
            status_slot.empty()
            st.rerun()
        else:
            status_slot.warning("Couldn't catch that — please speak again.")
            set_agent_ui_state("idle", "Chef is Ready — Tap microphone below to speak")

    # Insecure context warning for remote VM access over plain HTTP
    st.markdown("""
    <div id="mic-security-notice" style="display:none; margin: 6px 0 12px 0; padding: 10px 14px; background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 12px; font-size: 13px; color: #fca5a5;">
      🔒 <strong>Microphone Blocked by Browser:</strong> Chrome and Edge require <strong>HTTPS</strong> to access the microphone on remote servers. If testing via plain HTTP/IP, enable this URL in <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code>.
    </div>
    <script>
    if (!window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
      var el = document.getElementById('mic-security-notice');
      if (el) el.style.display = 'block';
    }
    </script>
    """, unsafe_allow_html=True)

    # Unified Browser Voice Interface (Seamless Real-time Voice Intelligence)
    mic_idx = st.session_state.get("mic_key_idx", 0)
    browser_audio = st.audio_input(
        "🎙️ Tap to Speak (Browser Microphone)",
        key=f"browser_mic_{mic_idx}"
    )
    if browser_audio is not None:
        audio_bytes = browser_audio.getvalue()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        if audio_hash != st.session_state.get("_last_processed_audio_hash"):
            st.session_state._last_processed_audio_hash = audio_hash
            st.session_state.mic_key_idx = mic_idx + 1
            set_agent_ui_state("user_speaking", "Hearing your voice... (Processing)")
            try:
                raw_data, sr = sf.read(io.BytesIO(audio_bytes))
                if raw_data.ndim > 1:
                    raw_data = raw_data.mean(axis=1)
                if sr != 16000:
                    raw_data = scipy.signal.resample(raw_data, int(len(raw_data) * 16000 / sr))
                handle_voice_audio(raw_data.astype(np.float32), source="browser")
            except Exception as e:
                st.error(f"Error reading browser audio: {e}")

    # Real-time Subtitles Area (High-Contrast, Clean & Large)
    st.markdown("<h3 style='margin-top: 24px; margin-bottom: 12px; color: #f8fafc;'>💬 Live Subtitles</h3>", unsafe_allow_html=True)

    # Customer Subtitle
    if st.session_state.customer_subtitle:
        st.markdown(f"""
        <div class="sub-card customer">
            <div class="sub-header">
                <span class="sub-badge-cust">🗣️ CUSTOMER</span>
                <span class="sub-time">{getattr(st.session_state, 'customer_time', '')}</span>
            </div>
            <div class="sub-text">"{st.session_state.customer_subtitle}"</div>
        </div>
        """, unsafe_allow_html=True)

    # Chef Subtitle
    if st.session_state.chef_subtitle:
        st.markdown(f"""
        <div class="sub-card chef">
            <div class="sub-header">
                <span class="sub-badge-chef">👨‍🍳 CHEF</span>
                <span class="sub-time">{getattr(st.session_state, 'chef_time', '')}</span>
            </div>
            <div class="sub-text">"{st.session_state.chef_subtitle}"</div>
        </div>
        """, unsafe_allow_html=True)

    # Helper for quick demo queries (clean, no repeating)
    def send_demo_query(query_text):
        st.session_state.customer_subtitle = query_text
        st.session_state.customer_time = time.strftime("%H:%M:%S")
        set_agent_ui_state("checking", "Agent is checking the menu & orders...")
        reply = st.session_state.agent.process_turn(query_text)
        st.session_state.chef_subtitle = reply
        st.session_state.chef_time = time.strftime("%H:%M:%S")
        if getattr(st.session_state.cart, "last_placed_order", None) and st.session_state.cart.is_submitted:
            st.session_state.placed_order_info = st.session_state.cart.last_placed_order
            st.session_state.last_order_num = st.session_state.placed_order_info.get("order_number")
            st.session_state.show_order_modal = True
            st.session_state.cart.is_submitted = False
        try:
            synth = st.session_state.tts.synthesize_audio(reply)
            if synth is not None:
                audio_out, sr_out = synth
                wav_buf = io.BytesIO()
                sf.write(wav_buf, audio_out, sr_out, format="WAV")
                st.session_state.browser_speech_wav = wav_buf.getvalue()
            else:
                st.session_state.tts.speak(reply)
        except Exception:
            st.session_state.tts.speak(reply)
        st.session_state.agent_mode = "speaking"
        st.session_state.agent_status = "Chef speaking aloud..."
        st.rerun()

    # Quick Demo Voice Queries for Easy Testing
    st.markdown("<p style='font-size: 13px; color: #94a3b8; margin-top: 18px; margin-bottom: 8px;'>Demo sample queries:</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🍨 See Soft Ice", use_container_width=True):
            send_demo_query("I want to see all items in soft ice")
    with c2:
        if st.button("🥤 Latte Float", use_container_width=True):
            send_demo_query("Can I get one Cafe Latte Float with less ice?")
    with c3:
        if st.button("☕ Hot Drinks", use_container_width=True):
            send_demo_query("get me the options of hot drinks")
    with c4:
        if getattr(st.session_state.cart, "confirmation_requested", False):
            if st.button("✅ Yes, Place It!", type="primary", use_container_width=True):
                send_demo_query("yes")
        else:
            if st.button("🛍️ Place Order", use_container_width=True):
                send_demo_query("place the order")

# ─────────────────────────────────────────────────────────────────────────────
# Right: Active Order Tray (Cart is stored in session, persists on refresh)
# ─────────────────────────────────────────────────────────────────────────────
with col_right:
    st.markdown("<h2 style='margin-bottom: 2px; color: #ffffff;'>🛒 Active Order Tray</h2>", unsafe_allow_html=True)
    st.caption("Items in your cart are preserved across page refreshes.")
    
    cart = st.session_state.cart
    
    with st.container(border=True):
        if cart.is_empty():
            st.markdown("""
            <div style="text-align: center; padding: 40px 10px; color: #94a3b8;">
                <div style="font-size: 40px; margin-bottom: 8px;">🛍️</div>
                <p style="font-size: 16px; font-weight: 700; color: #e2e8f0;">Your tray is currently empty.</p>
                <span style="font-size: 13px;">Speak to Chef or tap '+ Add' in the Quick Menu sidebar to begin!</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            for item in cart.items:
                col_name, col_meta = st.columns([3, 1.2])
                with col_name:
                    st.markdown(f"**{item.name}**")
                    if getattr(item, "addons", None):
                        for a in item.addons:
                            st.markdown(f"<div style='font-size: 13px; color: #38bdf8; margin-left: 6px;'>+ {a.get('name')} (+{float(a.get('price', 0.5)):.3f} BHD)</div>", unsafe_allow_html=True)
                    if item.notes and not getattr(item, "addons", None):
                        st.caption(f"Note: {item.notes}")
                with col_meta:
                    st.markdown(f"<div style='text-align: right; font-weight: 700; color: #34d399;'>{item.total_price:.3f} BHD</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: right; font-size: 13px; color: #94a3b8;'>Qty: {item.quantity}</div>", unsafe_allow_html=True)
                st.divider()
            
            # Subtotal & Item Count
            subtotal = cart.get_total()
            total_items = sum(i.quantity for i in cart.items)
            
            st.markdown(f"""
            <div style="background: #0f172a; border: 1px solid #1f2937; border-radius: 12px; padding: 14px 18px; display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                <div>
                    <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Total Items</span>
                    <div style="font-size: 16px; font-weight: 700; color: #ffffff;">{total_items} {'item' if total_items == 1 else 'items'}</div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Order Total</span>
                    <div style="font-size: 26px; font-weight: 900; color: #34d399;">{subtotal:.3f} BHD</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_clear, col_place = st.columns([1, 1.6])
            with col_clear:
                if st.button("🗑️ Clear Tray", use_container_width=True):
                    cart.clear()
                    st.session_state.customer_subtitle = None
                    st.session_state.chef_subtitle = "Your order tray has been cleared. What would you like to start with?"
                    st.session_state.chef_time = time.strftime("%H:%M:%S")
                    st.rerun()
                    
            with col_place:
                if not getattr(cart, "confirmation_requested", False):
                    if st.button("🛍️ Place Order", type="primary", use_container_width=True):
                        st.session_state.customer_subtitle = "Place the order"
                        st.session_state.customer_time = time.strftime("%H:%M:%S")
                        reply = st.session_state.agent.process_turn("place the order")
                        st.session_state.chef_subtitle = reply
                        st.session_state.chef_time = time.strftime("%H:%M:%S")
                        st.session_state.tts.speak(reply)
                        st.rerun()
                else:
                    if st.button("✅ Yes, Confirm & Place", type="primary", use_container_width=True):
                        st.session_state.customer_subtitle = "Yes, please place the order"
                        st.session_state.customer_time = time.strftime("%H:%M:%S")
                        reply = st.session_state.agent.process_turn("place the order")
                        st.session_state.chef_subtitle = reply
                        st.session_state.chef_time = time.strftime("%H:%M:%S")
                        if getattr(cart, "last_placed_order", None) and cart.is_submitted:
                            st.session_state.placed_order_info = cart.last_placed_order
                            st.session_state.last_order_num = st.session_state.placed_order_info.get("order_number")
                            st.session_state.show_order_modal = True
                            cart.is_submitted = False
                        st.session_state.tts.speak(reply)
                        st.rerun()

    # Confirmed Order Ticket Display
    if st.session_state.get("placed_order_info"):
        p_info = st.session_state.placed_order_info
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.12); border: 2px solid #10b981; border-radius: 14px; padding: 18px 20px; margin-top: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 12px; font-weight: 800; color: #34d399; letter-spacing: 1px; text-transform: uppercase;">✅ Order Placed Successfully</span>
                <span style="font-size: 12px; color: #94a3b8;">{p_info.get('time', '')}</span>
            </div>
            <div style="font-size: 30px; font-weight: 900; color: #ffffff; letter-spacing: 2px;">#{p_info.get('order_number')}</div>
            <div style="font-size: 15px; color: #cbd5e1; margin-top: 6px;">Total Amount: <strong style="color: #34d399; font-size: 18px;">{float(p_info.get('total', 0.0)):.3f} BHD</strong></div>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 2px;">Items: {len(p_info.get('items', []))} items sent to kitchen</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📄 View Placed Order Receipt", use_container_width=True):
            st.session_state.show_order_modal = True
            st.rerun()
