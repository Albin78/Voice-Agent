from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich import box
from typing import Optional
from cart import OrderCart

console = Console()


class KioskUI:
    """Rich visual console interface for restaurant kiosk."""

    @staticmethod
    def print_banner(restaurant_name: str, model_name: str, mode: str):
        console.clear()
        banner_text = Text()
        banner_text.append("🍽️  ", style="bold yellow")
        banner_text.append(f"{restaurant_name.upper()} KIOSK VOICE AGENT", style="bold cyan")
        banner_text.append("  |  ", style="dim")
        banner_text.append("Chef Voice Assistant", style="bold green")
        banner_text.append("\n", style="dim")
        banner_text.append("• Model: ", style="bold")
        banner_text.append(f"{model_name} (Local Ollama)", style="yellow")
        banner_text.append("   • VAD: ", style="bold")
        banner_text.append("Silero VAD v5 (Local 16kHz)", style="magenta")
        banner_text.append("   • Mode: ", style="bold")
        banner_text.append(f"{mode.upper()}", style="green")

        console.print(Panel(
            banner_text,
            box=box.DOUBLE,
            border_style="cyan",
            title="[bold yellow]Restaurant Kiosk System[/bold yellow]"
        ))

    @staticmethod
    def print_cart(cart: OrderCart):
        if cart.is_empty():
            cart_content = Text("Your cart is empty. What would you like to order today?", style="dim italic")
        else:
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
            table.add_column("Item", style="bold white")
            table.add_column("Qty", justify="center", style="yellow")
            table.add_column("Price", justify="right", style="green")
            table.add_column("Total", justify="right", style="bold green")

            for item in cart.items:
                note_str = f" [dim]({item.notes})[/dim]" if item.notes else ""
                table.add_row(
                    f"{item.name}{note_str}",
                    str(item.quantity),
                    f"${item.price:.2f}",
                    f"${item.total_price:.2f}"
                )

            total_table = Table.grid(expand=True)
            total_table.add_column()
            total_table.add_column(justify="right")
            total_table.add_row(
                Text(f"Total Items: {sum(i.quantity for i in cart.items)}", style="dim"),
                Text(f"Order Total: ${cart.get_total():.2f}", style="bold bright_green")
            )

            cart_content = table

        console.print(Panel(
            cart_content,
            box=box.ROUNDED,
            border_style="yellow",
            title=f"[bold yellow]🛒 Active Order ({len(cart.items)} items)[/bold yellow]",
            subtitle=f"[bold green]Total: ${cart.get_total():.2f}[/bold green]" if not cart.is_empty() else None
        ))

    @staticmethod
    def print_turn(speaker: str, text: str, is_chef: bool = False):
        if is_chef:
            panel = Panel(
                Text(text, style="bold white"),
                title="[bold green]👨‍🍳 Chef (Spoken Aloud)[/bold green]",
                border_style="green",
                box=box.ROUNDED
            )
        else:
            panel = Panel(
                Text(text, style="italic cyan"),
                title="[bold cyan]🗣️ Customer[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )
        console.print(panel)

    @staticmethod
    def print_status(message: str, style: str = "bold yellow"):
        console.print(f"[{style}]● {message}[/{style}]")
