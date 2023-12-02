# modules/menus/adaptanet_menu.py
from modules.menus import training_menu, testing_menu

def main():
    while True:
        print("[1] Training")
        print("[2] Testing")
        print("[3] Return to Main Menu")

        choice = input("Enter your choice: ")

        if choice == '1':
            training_menu.main()
        elif choice == '2':
            testing_menu.main()
        elif choice == '3':
            break
        else:
            print("Invalid choice, please try again.")

# modules/menus/training_menu.py
def main():
    print("Training Menu")
    # Implement training functionality here
