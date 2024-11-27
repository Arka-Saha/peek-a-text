
# Peek-A-Text Tool

A Python-based tool that creatively places user-provided text behind a person in an image. This tool utilizes **OpenCV**, **Pillow**, **NumPy**, and **MediaPipe** to make thumbnails, posters, and other visual designs stand out. 

## 🎨 Features
- Automatically detects a person in the image.
- Places user-provided text behind the person, creating a stunning overlapping effect.
- Supports creative designs for thumbnails, posters, and more.

## 🛠️ Technologies Used
- **OpenCV**: For image processing and manipulation.
- **Pillow (PIL)**: To handle text rendering.
- **NumPy**: For efficient matrix and image array operations.
- **MediaPipe**: To detect human subjects in the image.

## 📸 Example
### Input
An image of a person with no text.  
![Example Output](img/pic.jpg)
<br>
*(Sample Image)*
### Output
The same image with user-provided text placed behind the person.  
![Example Output](img/output.png) 
<br>
*(Sample Image)*  

## 🚀 How It Works
1. Detects the person in the image using **MediaPipe**.
2. Masks the person and places the text behind them.
3. Combines the text with the original image for the final effect.

## 📂 Project Structure
```
├── main.py            # Main script to run the tool
├── requirements.txt   # List of dependencies
├── README.md          # Project documentation
└── examples/          # Folder for input and output examples
```

## 🛠️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/peek-a-text.git
   cd text-behind-subject
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the tool:
   ```bash
   python main.py
   ```

## 💻 Usage
1. Provide the path to the input image.
2. Enter the desired text to place behind the person.
3. The output image is saved in the specified folder.

## 🌟 Future Enhancements
- Support for multiple subjects in an image.
- Add customization options for text style and placement.
- Build a GUI for non-technical users.

## 🤝 Contributing
Contributions are welcome! If you'd like to improve this project:
1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Open a pull request.

## 📧 Contact
If you have any questions or suggestions, feel free to reach out:
- **Arka Saha**
- [Your Email](mailto:i.am.arka.saha@gmail.com)
- [Your LinkedIn Profile](http://linkedin.com/in/arka-saha2707/)
