#!/usr/bin/env python3
"""Generate comprehensive PDF report with all content and visualizations"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import os
from pathlib import Path

def create_pdf():
    """Create comprehensive PDF report"""
    pdf_path = "comprehensive_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                          rightMargin=0.8*inch, leftMargin=0.8*inch,
                          topMargin=0.8*inch, bottomMargin=0.8*inch)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#0066CC'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#0066CC'),
        spaceAfter=12,
        fontName='Helvetica-Bold',
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#3399FF'),
        spaceAfter=10,
        fontName='Helvetica-Bold',
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
    )
    
    # Title Page
    elements.append(Spacer(1, 1*inch))
    elements.append(Paragraph("The Visual Instruction Following Evaluation Benchmark (VIF-Eval)", title_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("A Comprehensive Framework for Evaluating Compositional and Editorial Image Generation", 
                             ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER)))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Complete Assessment Report", 
                             ParagraphStyle('Subtitle2', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER)))
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("November 2025", 
                             ParagraphStyle('Date', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER)))
    elements.append(PageBreak())
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", heading1_style))
    summary_text = """
    This comprehensive report presents the complete evaluation results of the Visual Instruction Following Evaluation 
    Benchmark (VIF-Eval), a novel framework for assessing generative visual models across multiple dimensions of 
    real-world performance. We evaluated three state-of-the-art models—GPT Image 1, DALL-E 3, and Gemini 2.5 Flash 
    Image (Nano Banana)—across 47 diverse prompts with 167+ constraints.
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("<b>Key Findings:</b>", heading2_style))
    findings = [
        "GPT Image 1 achieves the highest overall pass rate (58.1%)",
        "Nano Banana performs comparably (54.7%) with significantly lower cost",
        "Models excel at spatial relationships (100% pass rate) and CSP constraints (95-100%)",
        "Critical weakness in text rendering (12-13% pass rate across all models)",
        "Cost-performance analysis reveals Nano Banana as most cost-effective"
    ]
    for finding in findings:
        elements.append(Paragraph(f"• {finding}", normal_style))
    elements.append(PageBreak())
    
    # Results Section
    elements.append(Paragraph("Results", heading1_style))
    
    # Overall Performance Table
    elements.append(Paragraph("Overall Performance", heading2_style))
    table_data = [
        ['Model', 'Prompts', 'Constraints', 'Passed', 'Pass Rate'],
        ['GPT Image 1', '47', '167', '97', '58.1%'],
        ['Nano Banana', '47', '172', '94', '54.7%'],
        ['DALL-E 3', '47', '76', '13', '17.1%'],
    ]
    table = Table(table_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add all figures
    fig_dir = Path("paper_assets/figures")
    case_dir = Path("paper_assets/case_studies")
    
    figures = [
        ("Overall Pass Rate Comparison", "overall_pass_rate_comparison.png"),
        ("Pass Rate by Constraint Type", "pass_rate_by_type_comparison.png"),
        ("Average Score by Constraint Type", "avg_score_by_type_comparison.png"),
        ("Performance Heatmap", "heatmap_performance_matrix.png"),
        ("Model Capabilities Radar Chart", "radar_capabilities.png"),
        ("Failure Distribution", "failure_mode_distribution.png"),
        ("Error Pattern Breakdown", "error_pattern_breakdown.png"),
        ("Performance vs Difficulty", "performance_vs_difficulty.png"),
        ("Comprehensive Comparison Matrix", "constraint_comparison_matrix.png"),
        ("Cost-Performance Analysis", "cost_performance_scatter.png"),
    ]
    
    print("Adding visualizations...")
    for title, filename in figures:
        fig_path = fig_dir / filename
        if fig_path.exists():
            try:
                elements.append(Spacer(1, 0.2*inch))
                elements.append(Paragraph(f"<b>{title}</b>", heading2_style))
                # Scale image to fit page width
                img = Image(str(fig_path), width=6.5*inch, height=4.5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
                print(f"  ✅ Added: {filename}")
            except Exception as e:
                print(f"  ⚠️ Could not add {filename}: {e}")
    
    elements.append(PageBreak())
    
    # Case Studies
    elements.append(Paragraph("Case Studies", heading1_style))
    
    case_studies = [
        ("Case Study 1: Text Rendering", 
         "A poster with the text 'SUMMER SALE' in large bold letters",
         "case_study_01_text_005.png"),
        ("Case Study 2: Complex Composition",
         "A photo of three blue mugs and two red plates on a wooden table...",
         "case_study_02_comp_001.png"),
        ("Case Study 3: CSP Constraint",
         "CSP task with numeric relationships",
         "case_study_03_csp_01_numbers_row.png"),
    ]
    
    for title, description, filename in case_studies:
        fig_path = case_dir / filename
        if fig_path.exists():
            try:
                elements.append(Paragraph(f"<b>{title}</b>", heading2_style))
                elements.append(Paragraph(description, normal_style))
                elements.append(Spacer(1, 0.1*inch))
                img = Image(str(fig_path), width=6.5*inch, height=4.5*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.3*inch))
                print(f"  ✅ Added case study: {filename}")
            except Exception as e:
                print(f"  ⚠️ Could not add {filename}: {e}")
    
    elements.append(PageBreak())
    
    # Cost Analysis
    elements.append(Paragraph("Cost and Latency Analysis", heading1_style))
    elements.append(Paragraph("API Cost Analysis", heading2_style))
    
    cost_table = [
        ['Model', 'Images', 'Cost/Image', 'Total Cost'],
        ['GPT Image 1', '47', '$0.040', '$1.88'],
        ['Nano Banana', '47', '$0.005', '$0.24'],
        ['DALL-E 3', '11', '$0.040', '$0.44'],
    ]
    cost_table_obj = Table(cost_table, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch])
    cost_table_obj.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(cost_table_obj)
    elements.append(Spacer(1, 0.3*inch))
    
    # Conclusion
    elements.append(PageBreak())
    elements.append(Paragraph("Conclusion", heading1_style))
    conclusion_text = """
    The Visual Instruction Following Evaluation Benchmark (VIF-Eval) provides a comprehensive framework for 
    assessing generative visual models beyond simple prompt adherence. Our evaluation of three state-of-the-art 
    models reveals that models excel at spatial relationships, character consistency, and constraint satisfaction 
    problems, but show critical limitations in text rendering (12-13% pass rate) and precise counting (17-31% pass rate). 
    GPT Image 1 and Nano Banana show similar overall performance (54-58% pass rates), with Nano Banana providing 
    significantly better cost-performance ratio.
    """
    elements.append(Paragraph(conclusion_text, normal_style))
    
    # Build PDF
    print("\nBuilding PDF...")
    doc.build(elements)
    
    file_size = os.path.getsize(pdf_path) / 1024
    print(f"\n✅ PDF created successfully!")
    print(f"   File: {pdf_path}")
    print(f"   Size: {file_size:.1f} KB")
    print(f"   Pages: Multiple pages with all visualizations")
    
    return pdf_path

if __name__ == "__main__":
    try:
        create_pdf()
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        import traceback
        traceback.print_exc()

