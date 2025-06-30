# 🚀 ChatGPT Analyzer - Quick Start Guide

Welcome! This guide will get you analyzing your ChatGPT conversations in just a few minutes.

## 🎯 The Easiest Way: Interactive Mode

**No command-line arguments needed!** Just run the script and answer simple questions.

### Step 1: Install Dependencies

**Windows:**
```batch
install_windows.bat
```

**Mac/Linux:**
```bash
chmod +x install_deps.sh
./install_deps.sh
```

### Step 2: Run the Analyzer

```bash
python gptscraper.py
```

That's it! The script will ask you everything it needs to know.

## 📋 What You'll Be Asked

### Question 1: Choose Your Mode
```
How would you like to analyze your ChatGPT conversations?
  1. Export mode (analyze downloaded JSON file)
  2. Live mode (browse ChatGPT in real-time)

Answer: 1
```

**Tip:** Choose Export mode for your first time - it's more reliable!

### Question 2: File Location (Export Mode)
```
What's the path to your conversations.json file?
(Press Enter for default: conversations.json)

Answer: [Just drag and drop your file here!]
```

### Question 3: Output Format
```
Select format option:
  1. HTML only (beautiful interactive report)
  2. HTML + CSV (report + spreadsheet)
  3. All formats (HTML, CSV, JSON, Markdown, PDF)
  4. Custom selection

Answer: 1
```

**Tip:** HTML is perfect for most users - it's beautiful and interactive!

### Question 4: Extract Content?
```
Would you like to extract code and research content from conversations?
  1. Yes
  2. No
  3. Ask me after analysis

Answer: 3
```

## 📥 Getting Your ChatGPT Data (For Export Mode)

1. Go to [ChatGPT](https://chat.openai.com/)
2. Click your profile → Settings
3. Go to "Data Controls"
4. Click "Export data"
5. Check your email (arrives in ~10 minutes)
6. Download and extract the ZIP file
7. Find `conversations.json` inside

## 🎉 That's It!

After answering the questions:
- The analyzer will process your conversations
- Generate beautiful reports
- Open the HTML report in your browser automatically
- Save everything to the `chatgpt_analysis` folder

## 💡 Pro Tips

### See What Each Conversation Contains
The HTML report shows:
- 💻 Code conversations
- 🔬 Research discussions  
- 🔗 Conversations with links
- 📊 Statistics and charts

### Extract Code Files
After analysis, you can extract:
- All code snippets as proper files (.py, .js, etc.)
- Research conversations as markdown documents
- Complete conversation transcripts

### Filter and Search
The HTML report lets you:
- Filter by type (code, research, mixed)
- Search by title or keywords
- See language distribution
- Track your ChatGPT usage patterns

## 🆘 Need Help?

### Common Issues

**"File not found"**
- Make sure you extracted the ZIP file
- The file might be in your Downloads folder
- Try dragging the file directly into the terminal

**"Chrome not found"** 
- Export mode doesn't need Chrome!
- Only Live mode requires Chrome browser

**"Package installation failed"**
- Try running the installer as administrator (Windows)
- On Mac/Linux, you might need: `sudo pip install -r requirements.txt`

### Still Stuck?

1. Check the [Full Documentation](README.md)
2. Run the installation test: `python test_installation.py`
3. Enable debug mode when the script asks about advanced options

## 🎯 Quick Examples

### Just Analyze Everything
```bash
python gptscraper.py
# Choose: Export mode
# Enter: path/to/conversations.json
# Choose: HTML only
# Done!
```

### Extract All Code
```bash
python gptscraper.py
# Choose: Export mode
# Enter: path/to/conversations.json
# Choose: HTML only
# Choose: Yes (extract content)
```

### See What's Happening (Live Mode)
```bash
python gptscraper.py
# Choose: Live mode
# Enter: your email
# Enter: your password
# Choose: No (see what's happening)
```

---

**Remember:** The interactive mode guides you through everything. No need to memorize commands!

Happy analyzing! 🎉
