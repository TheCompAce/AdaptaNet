from rich.console import Console
from modules.menus import settings_menu, adaptanet_menu

# Initialize Rich Console for coloring
console = Console()

def main_menu():
    while True:
        console.print("[1] Settings", style="bold cyan")
        console.print("[2] AdaptaNet", style="bold cyan")
        console.print("[3] Exit", style="bold cyan")

        choice = input("Enter your choice: ")

        if choice == '1':
            settings_menu.main()
        elif choice == '2':
            adaptanet_menu.main()
        elif choice == '3':
            console.print("Exiting AdaptaNet...", style="bold red")
            break
        else:
            console.print("Invalid choice, please try again.", style="bold yellow")

if __name__ == "__main__":
    console.print("Welcome to AdaptaNet", style="bold green")
    main_menu()
