# Quick Start: Tab Chaining

## 🎯 Goal
Process audio files and analyze them for stock mentions **without copying/pasting** between tabs.

## 📋 3-Step Process

### Step 1: Transcribe Audio
```
📁 Go to "3️⃣ Auto-Diarize & Transcribe" tab
   ↓
📤 Upload your audio file(s)
   ↓
⚙️ Configure settings (optional)
   ↓
🚀 Click "Transcribe Audio"
   ↓
⏳ Wait for completion
   ↓
✅ JSON output appears at bottom
```

### Step 2: Load Data
```
📁 Go to "🔟 JSON Batch Analysis" tab
   ↓
📥 Click "Load from STT Tab" button
   ↓
✅ JSON automatically fills the input box
```

### Step 3: Analyze
```
⚙️ Configure LLM settings (optional)
   ↓
🚀 Click "Analyze All Conversations"
   ↓
✅ Stock extraction results appear
```

## 🎬 Example

```
Input:  audio_call.wav (3 minutes)
         ↓
Step 1:  STT Tab processes → Generates transcription
         ↓
Step 2:  Click "Load from STT Tab" → Data auto-loads
         ↓
Step 3:  Analyze → Extracts: "騰訊 (00700)", "阿里巴巴 (09988)"
```

## 💡 Key Points

- ✅ **No Copy/Paste**: Data flows automatically
- ✅ **Multiple Files**: Process multiple audio files at once
- ✅ **Both Models**: Includes SenseVoice and Whisper-v3 results
- ✅ **Complete Data**: Metadata, timestamps, everything preserved
- ✅ **Still Manual Works**: You can still paste JSON manually if needed

## ⚠️ Troubleshooting

**Q: "Load from STT Tab" shows "No data from STT tab"?**  
A: Run the STT tab first and wait for it to complete.

**Q: Old data appears?**  
A: Re-run STT tab to update the data.

**Q: Can I still paste JSON manually?**  
A: Yes! The manual input still works perfectly.

## 🔧 Technical Note

Under the hood:
- Uses Gradio's `gr.State()` for data sharing
- JSON format matches exactly what JSON Batch Analysis expects
- State persists for the entire session
- No data is saved to disk unless you export it

## 📚 More Info

- See `TAB_CHAINING_GUIDE.md` for detailed documentation
- See `CHAINING_SUMMARY.md` for technical implementation details

---

**That's it!** You're ready to use tab chaining. 🎉

