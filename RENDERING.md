# GitHub Rendering Guide

This document explains how to ensure proper rendering of HLD.md on GitHub.

## Mermaid Diagrams

GitHub automatically renders Mermaid diagrams in markdown files. The HLD.md file contains several Mermaid diagrams that will render automatically when viewed on GitHub.

### Viewing on GitHub

1. **Direct View**: Open `HLD.md` on GitHub - diagrams render automatically
2. **Raw View**: Click "Raw" to see the source code
3. **Preview**: Use GitHub's preview feature in pull requests

### If Diagrams Don't Render

1. **Check Browser**: Ensure you're using a modern browser (Chrome, Firefox, Safari, Edge)
2. **Refresh**: Hard refresh the page (Ctrl+F5 or Cmd+Shift+R)
3. **Check Syntax**: Mermaid code blocks must use ` ```mermaid ` (not just ` ``` `)

### Testing Locally

To preview how it will look on GitHub:

1. **VS Code**: Install "Markdown Preview Mermaid Support" extension
2. **Online**: Use https://mermaid.live/ to test individual diagrams
3. **GitHub Desktop**: Preview feature shows rendered markdown

## File Structure

```
vis_ifeval/
├── HLD.md              ← Main HLD document (renders on GitHub)
├── ARCHITECTURE.md     ← Technical architecture (renders on GitHub)
├── README.md           ← Project documentation (renders on GitHub)
└── .github/
    └── workflows/
        └── render-check.yml  ← GitHub Actions workflow
```

## Mermaid Diagram Types Used

1. **Flowcharts** (`graph TB`, `graph LR`) - System flow
2. **Sequence Diagrams** (`sequenceDiagram`) - Process flow
3. **Component Diagrams** - Architecture components

All diagrams use standard Mermaid syntax compatible with GitHub.

## Troubleshooting

### Issue: Diagrams show as code blocks
**Solution**: Ensure code blocks use ` ```mermaid ` (with "mermaid" language identifier)

### Issue: Diagrams are cut off
**Solution**: GitHub has size limits. Break large diagrams into smaller ones.

### Issue: Special characters not rendering
**Solution**: Use HTML entities or plain text in node labels.

## Verification

To verify rendering:
1. Push to GitHub
2. Open the file in the web interface
3. Diagrams should render automatically
4. If not, check browser console for errors

---

**Last Updated**: November 2024

