import argparse
import sys
import logging
import time
from config import Config
from cart import OrderCart
from menu_api import MenuAPIClient
from agent import ChefAgent
from vad import SileroVADDetector
from stt import WhisperSTT
from tts import TextToSpeech
from kiosk_ui import KioskUI, console

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Main")


def run_text_mode(agent: ChefAgent, tts: TextToSpeech, enable_tts: bool = True):
    """Interactive text simulation mode for testing conversation and tools."""
    KioskUI.print_banner(Config.RESTAURANT_NAME, Config.OLLAMA_MODEL, mode="Text Simulation")

    # Chef initial greeting
    greeting = agent.get_greeting()
    KioskUI.print_cart(agent.cart)
    KioskUI.print_turn("Chef", greeting, is_chef=True)
    if enable_tts:
        tts.speak(greeting)

    console.print("[dim]Type what the customer says (or 'exit' / 'quit' to stop):[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]Customer > [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Exiting kiosk session. Goodbye![/yellow]")
                break

            KioskUI.print_turn("Customer", user_input, is_chef=False)
            KioskUI.print_status("Chef is checking menu & preparing response...", "yellow")

            # Process turn through Chef agent
            reply = agent.process_turn(user_input)

            # Update UI
            KioskUI.print_cart(agent.cart)
            KioskUI.print_turn("Chef", reply, is_chef=True)

            # Speak reply aloud
            if enable_tts:
                tts.speak(reply)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended.[/yellow]")
            break


def run_voice_mode(agent: ChefAgent, vad: SileroVADDetector, stt: WhisperSTT, tts: TextToSpeech):
    """Full voice ordering mode using Silero VAD v5 and Whisper STT."""
    KioskUI.print_banner(Config.RESTAURANT_NAME, Config.OLLAMA_MODEL, mode="Live Voice Kiosk")

    # Chef initial greeting
    greeting = agent.get_greeting()
    KioskUI.print_cart(agent.cart)
    KioskUI.print_turn("Chef", greeting, is_chef=True)
    tts.speak(greeting)

    console.print("\n[bold green]🎤 Silero VAD v5 is active. Speak into your microphone...[/bold green]")
    console.print("[dim]Press Ctrl+C anytime to stop.[/dim]\n")

    while True:
        try:
            def on_speech_start():
                KioskUI.print_status("🎙️  Speech detected! Recording...", "bright_green")

            # Listen for speech using Silero VAD v5
            audio = vad.listen_and_record_speech(
                on_speech_start=on_speech_start,
                on_speech_end=None
            )

            if audio is None:
                # Ambient noise / breath / click filtered out by VAD
                time.sleep(0.05)
                continue

            KioskUI.print_status("🤫 Processing speech with Whisper...", "bright_blue")

            # Transcribe audio with Whisper
            transcript = stt.transcribe(audio)
            if not transcript:
                time.sleep(0.05)
                continue

            KioskUI.print_turn("Customer", transcript, is_chef=False)
            KioskUI.print_status("🧠 Chef thinking...", "yellow")

            # Process turn
            reply = agent.process_turn(transcript)

            # Display updated cart & Chef's response
            KioskUI.print_cart(agent.cart)
            KioskUI.print_turn("Chef", reply, is_chef=True)

            # Speak Chef's response aloud
            tts.speak(reply)

            # Short acoustic settling delay to prevent microphone from picking up speaker echo
            time.sleep(0.5)
            console.print("\n[bold green]🎤 Listening for your next response...[/bold green]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Voice kiosk session closed.[/yellow]")
            break
        except Exception as e:
            logger.error(f"Error in voice loop: {e}")
            time.sleep(1)


def test_api():
    """Test connection to Categories and Products APIs."""
    console.print("[bold yellow]Testing API connections...[/bold yellow]")
    client = MenuAPIClient(force_mock=False)

    console.print(f"Categories URL: {Config.PRODUCTS_CATEGORY_URL}")
    cat_res = client.get_categories()
    if cat_res.get("success"):
        console.print(f"[green]✓ Categories API OK:[/green] Found {len(cat_res.get('categories', []))} categories.")
        for c in cat_res.get("categories", [])[:5]:
            console.print(f"  • ID {c.get('categoryId')}: {c.get('categoryName')}")
    else:
        console.print(f"[red]✗ Categories API Failed:[/red] {cat_res.get('error')}")

    console.print(f"\nProducts API: {Config.PRODUCTS_API}")
    prod_res = client.get_products()
    if prod_res.get("success"):
        console.print(f"[green]✓ Products API OK:[/green] Found {len(prod_res.get('products', []))} products.")
    else:
        console.print(f"[red]✗ Products API Failed:[/red] {prod_res.get('error')}")


def test_audio():
    """Test audio recording, Silero VAD detection, and TTS speaker."""
    console.print("[bold yellow]1. Testing Text-To-Speech (pyttsx3)...[/bold yellow]")
    tts = TextToSpeech()
    tts.speak("Testing audio speaker. Text to speech is working properly.")
    console.print("[green]✓ Speaker test complete.[/green]")

    console.print("\n[bold yellow]2. Initializing Silero VAD v5...[/bold yellow]")
    vad = SileroVADDetector()
    console.print("[green]✓ Silero VAD loaded successfully.[/green]")

    console.print("\n[bold yellow]3. Please speak a sentence into your microphone now...[/bold yellow]")
    audio = vad.listen_and_record_speech(
        on_speech_start=lambda: console.print("[green]● Speech detected![/green]"),
        on_speech_end=lambda: console.print("[blue]● Silence detected, recording finished.[/blue]")
    )

    if audio is not None:
        console.print(f"[green]✓ Recorded {len(audio)/16000:.2f} seconds of audio.[/green]")
        console.print("\n[bold yellow]4. Transcribing with Whisper...[/bold yellow]")
        stt = WhisperSTT()
        text = stt.transcribe(audio)
        console.print(f"[bold cyan]Transcribed Text:[/bold cyan] \"{text}\"")
    else:
        console.print("[yellow]No speech was detected within timeout.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Kiosk Voice Agent - Chef Voice Assistant")
    parser.add_argument("--voice", action="store_true", help="Start in live voice mode with Silero VAD v5 and Whisper")
    parser.add_argument("--text", action="store_true", help="Start in interactive text mode")
    parser.add_argument("--test-audio", action="store_true", help="Test microphone, Silero VAD, and TTS")
    parser.add_argument("--test-api", action="store_true", help="Test categories and products API endpoints")
    parser.add_argument("--mock", action="store_true", help="Force realistic fallback menu for offline development")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS speech playback in text mode")
    parser.add_argument("--model", type=str, default=Config.OLLAMA_MODEL, help="Ollama model name (default: llama3.2:3b)")

    args = parser.parse_args()

    if args.test_api:
        test_api()
        return

    if args.test_audio:
        test_audio()
        return

    # Update config model if specified
    if args.model:
        Config.OLLAMA_MODEL = args.model

    # Initialize components
    cart = OrderCart()
    menu_client = MenuAPIClient(force_mock=args.mock)
    agent = ChefAgent(cart=cart, menu_client=menu_client)
    tts = TextToSpeech()

    if args.text:
        # Interactive text simulation mode (customer types in terminal)
        run_text_mode(agent, tts, enable_tts=not args.no_tts)
    else:
        # Default: Live voice mode (customer speaks into microphone)
        vad = SileroVADDetector()
        stt = WhisperSTT()
        run_voice_mode(agent, vad, stt, tts)


if __name__ == "__main__":
    main()
