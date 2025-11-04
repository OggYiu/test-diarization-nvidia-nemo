# Enhanced Format Feature - Example Output

## Example Scenario

**Audio File**: `[Dickson Lau 0489]_8330-97501167_20251020100120(20981).wav`

**RTTM Content**:
```
SPEAKER test 1   0.380   0.875 <NA> <NA> speaker_1 <NA> <NA>
SPEAKER test 1   1.255   4.705 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER test 1   5.960   3.200 <NA> <NA> speaker_1 <NA> <NA>
```

**LLM Identification**: speaker_0 = 經紀 (Broker)

---

## Output Comparison

### WITHOUT Enhanced Format (Default Behavior)
**SenseVoiceSmall Result Textbox:**
```
經紀 Dickson Lau: 请到时点啊。
客戶 CHENG SUK HING: 刘生啊，我想买腾讯个轮啊买个声得唔得啊嗯。
經紀 Dickson Lau: 好的，我帮你下单。
```

**Whisper-v3-Cantonese Result Textbox:**
```
經紀 Dickson Lau: 请到时点啊
客戶 CHENG SUK HING: 刘生啊我想买腾讯个轮啊买个声得唔得啊
經紀 Dickson Lau: 好的我帮你下单
```

---

### WITH Enhanced Format (Checkbox Enabled) ✨
**SenseVoiceSmall (經紀/客戶) Result Textbox:**
```
對話時間: 2025-10-20T18:01:20
經紀: Dickson Lau
broker_id: 0489
客戶: CHENG SUK HING
client_id: P77197

- 經紀 Dickson Lau (1.255): 请到时点啊。
- 客戶 CHENG SUK HING (0.380): 刘生啊，我想买腾讯个轮啊买个声得唔得啊嗯。
- 經紀 Dickson Lau (5.960): 好的，我帮你下单。
```

**Whisper-v3-Cantonese (經紀/客戶) Result Textbox:**
```
對話時間: 2025-10-20T18:01:20
經紀: Dickson Lau
broker_id: 0489
客戶: CHENG SUK HING
client_id: P77197

- 經紀 Dickson Lau (1.255): 请到时点啊
- 客戶 CHENG SUK HING (0.380): 刘生啊我想买腾讯个轮啊买个声得唔得啊
- 經紀 Dickson Lau (5.960): 好的我帮你下单
```

---

## Key Features Demonstrated

### 1. Metadata Header
```
對話時間: 2025-10-20T18:01:20    ← Converted from UTC to HKT (UTC+8)
經紀: Dickson Lau                  ← Extracted from filename
broker_id: 0489                    ← Extracted from filename
客戶: CHENG SUK HING               ← Looked up from client.csv
client_id: P77197                  ← Looked up from client.csv
```

### 2. Timestamp Format
```
- 經紀 Dickson Lau (1.255): 请到时点啊。
  ↑              ↑     ↑
  Role         Time   Text
               (seconds from start)
```

### 3. Speaker Order
- The order is based on RTTM segments, NOT the speaker labels
- In this example, speaker_1 speaks first at 0.380s
- Then speaker_0 speaks at 1.255s
- The timestamps correctly reflect when each person started speaking

---

## How to Use

1. **Open the "3️⃣ Auto-Diarize & Transcribe" tab**

2. **Upload your audio file**
   - File should follow the naming pattern with metadata

3. **Enable the enhanced format checkbox** ✅
   - Look for: "📋 Enhanced format (metadata + timestamps)"
   - Check the box to enable

4. **Select your models**
   - SenseVoiceSmall ✅
   - Whisper-v3-Cantonese ✅ (optional)

5. **Click "🎯 Auto-Diarize & Transcribe"**

6. **View the enhanced results**
   - Both textboxes will show metadata headers
   - Each line will have timestamps in parentheses
   - Format: `- Role Name (time): transcribed text`

---

## When to Use Enhanced Format

### Use Enhanced Format When:
✅ You need to know the exact timing of each utterance  
✅ You want quick access to conversation metadata  
✅ You're analyzing call recordings for compliance  
✅ You need to correlate transcriptions with other time-stamped data  
✅ You want a clean, structured format for further processing  

### Use Default Format When:
✅ You just want to read the conversation  
✅ You don't need timing information  
✅ You prefer a simpler, more compact output  
✅ You're doing quick transcriptions  

---

## Technical Notes

- **Time Format**: Seconds from the start of the audio (e.g., 0.380 = 380 milliseconds, 1.255 = 1.255 seconds)
- **Precision**: Timestamps are accurate to milliseconds based on RTTM diarization
- **Speaker Matching**: Timestamps are matched to transcription segments in order
- **Metadata Source**: All metadata comes from filename parsing and client.csv lookup
- **Time Zone**: All times are displayed in HKT (Hong Kong Time = UTC+8)
- **Character Encoding**: 
  - **SenseVoiceSmall**: Automatically converted to Traditional Chinese (繁體中文) using OpenCC
  - **Whisper-v3-Cantonese**: Already outputs Traditional Chinese

