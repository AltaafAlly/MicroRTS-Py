# 🎮 MicroRTS UTT Editor

A beautiful, user-friendly web-based editor for MicroRTS Unit Type Table (UTT) files. No more manually editing JSON files!

## ✨ Features

- **Visual Interface**: Clean, modern web interface for editing UTT files
- **Real-time Editing**: See changes instantly as you type
- **Unit Cards**: Each unit type displayed in an easy-to-read card format
- **Property Validation**: Input validation for numeric and boolean values
- **Array Support**: Easy editing of production lists and dependencies
- **Save & Export**: Save your changes as new UTT files
- **Reset Function**: Restore original values if needed
- **Drag & Drop**: Drag UTT files directly onto the editor

## 🚀 Quick Start

### Method 1: Using the Launcher (Recommended)
```bash
# List available UTT files
python scripts/utt_editor_launcher.py list

# Open editor with a specific UTT file
python scripts/utt_editor_launcher.py open CustomDemoUTT.json

# Open editor (load file manually)
python scripts/utt_editor_launcher.py open
```

### Method 2: Direct Browser Access
1. Open `scripts/utt_editor.html` in your web browser
2. Click "Load UTT File" and select your UTT file
3. Start editing!

## 📁 Available UTT Files

- `CustomDemoUTT.json` - Default balanced UTT
- `AsymmetricP1UTT.json` - Asymmetric UTT for Player 1
- `TestUnitTypeTable.json` - Test UTT file

## 🎯 How to Use

### 1. Loading a UTT File
- Click "📁 Load UTT File" button
- Select any `.json` UTT file from your project
- The editor will display all unit types in card format

### 2. Editing Unit Properties
Each unit card shows:
- **Basic Stats**: Cost, HP, Damage, Range, Sight
- **Timing**: Production, Move, Attack, Harvest times
- **Capabilities**: Checkboxes for what the unit can do
- **Production**: What it produces and what produces it

### 3. Global Settings
- **Move Conflict Resolution Strategy**: How units handle movement conflicts

### 4. Saving Your Changes
- Click "💾 Save UTT" to download your modified UTT
- Click "📤 Export JSON" for a clean JSON export
- Use "🔄 Reset" to restore original values

## 🔧 Unit Properties Explained

### Combat Properties
- **Cost**: Resources needed to produce this unit
- **HP**: Hit points (health)
- **Min/Max Damage**: Damage range when attacking
- **Attack Range**: How far the unit can attack
- **Sight Radius**: How far the unit can see

### Timing Properties
- **Produce Time**: Time to build this unit
- **Move Time**: Time to move one tile
- **Attack Time**: Time between attacks
- **Harvest Time**: Time to harvest resources
- **Return Time**: Time to return resources to base
- **Harvest Amount**: Resources gained per harvest

### Capabilities (Checkboxes)
- **Is Resource**: This unit is a resource node
- **Is Stockpile**: This unit can store resources
- **Can Harvest**: This unit can gather resources
- **Can Move**: This unit can move around
- **Can Attack**: This unit can attack enemies

### Production
- **Produces**: List of units this can build (comma-separated)
- **Produced By**: What can build this unit (comma-separated)

## 💡 Tips for UTT Design

### Balanced Gameplay
- Keep unit costs proportional to their power
- Ensure counter-units exist for each unit type
- Balance production times with unit effectiveness

### Asymmetric Design
- Give different players different unit types
- Vary costs, stats, or capabilities between players
- Create unique strategic options for each side

### Testing Changes
- Start with small changes to existing UTTs
- Test in actual matches to see the impact
- Use the tournament script to compare performance

## 🎮 Integration with Tournament Script

After editing your UTT files:

1. Save your modified UTT with a new name (e.g., `MyCustomUTT.json`)
2. Place it in `gym_microrts/microrts/utts/`
3. Use it in your tournament script:

```bash
# Edit the script to use your new UTT
python scripts/run_match_configured_with_events.py \
    --utt-p0 utts/MyCustomUTT.json \
    --utt-p1 utts/AsymmetricP1UTT.json \
    --interactive
```

## 🐛 Troubleshooting

### Editor Won't Load
- Make sure you're opening the HTML file in a modern browser
- Check that JavaScript is enabled
- Try refreshing the page

### File Won't Load
- Ensure the UTT file is valid JSON
- Check that the file has the required structure
- Try with one of the default UTT files first

### Changes Not Saving
- Make sure you click "Save UTT" after making changes
- Check that your browser allows file downloads
- Try the "Export JSON" option as an alternative

## 🔄 Workflow Example

1. **Start**: `python scripts/utt_editor_launcher.py open CustomDemoUTT.json`
2. **Edit**: Modify unit stats in the web interface
3. **Save**: Download your modified UTT file
4. **Test**: Use it in a tournament to see the effects
5. **Iterate**: Make more changes based on results

## 📝 File Structure

```
scripts/
├── utt_editor.html              # Main web editor
├── utt_editor_launcher.py       # Python launcher script
├── README_utt_editor.md         # This documentation
└── gym_microrts/microrts/utts/  # UTT files directory
    ├── CustomDemoUTT.json
    ├── AsymmetricP1UTT.json
    └── TestUnitTypeTable.json
```

---

**Happy UTT Editing!** 🎮✨

The visual editor makes it much easier to understand and modify unit properties without dealing with raw JSON. Perfect for experimenting with different unit configurations and asymmetric gameplay!
