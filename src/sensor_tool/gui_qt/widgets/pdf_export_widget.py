"""
PDF Export Widget for Modern Sensor Comparison Reports

This module provides comprehensive PDF export functionality using ReportLab
to generate standard sensor comparison reports with charts, tables, and analysis.
"""

import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.spider import SpiderChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

matplotlib.use("Agg")  # Non-interactive backend for PDF generation

from ...utils import (
    extract_numeric,
    extract_price_avg,
    extract_resolution,
    format_label,
)
from ..utils.font_manager import create_styled_font
from ..utils.theme_manager import get_theme_manager

logger = logging.getLogger(__name__)


def sanitize_language(text: str) -> str:
    """
    Sanitize superlative language from PDF report text to maintain neutral tone.

    Replaces overly positive language with neutral alternatives to ensure
    objective reporting suitable for technical documentation.
    """
    # Dictionary of problematic words/phrases and their neutral replacements
    replacements = {
        # Superlatives
        "excellent": "good",
        "superior": "notable",
        "outstanding": "notable",
        "exceptional": "notable",
        "premier": "notable",
        "best-in-class": "high-performing",
        # Comparative phrases - more contextual replacements
        "offers the best": "provides a good",
        "provides the best": "offers a good",
        "shows the best": "shows good",
        "demonstrates the best": "demonstrates good",
        "the best performance": "good performance",
        "best value": "good value",
        "optimal choice": "suitable choice",
        "ideal solution": "suitable solution",
        "ideal for": "suitable for",
        "perfect for": "suitable for",
        # Emphatic phrases
        "clearly superior": "notably different",
        "significantly better": "notably different",
        "far superior": "notably different",
    }

    sanitized_text = text
    for problematic, neutral in replacements.items():
        # Case-insensitive replacement while preserving original case
        import re

        pattern = re.compile(re.escape(problematic), re.IGNORECASE)
        sanitized_text = pattern.sub(neutral, sanitized_text)

    # Log replacements for debugging
    if sanitized_text != text:
        logger.debug(f"Language sanitization applied: '{text}' -> '{sanitized_text}'")

    return sanitized_text


class PDFGenerationThread(QThread):
    """Background thread for PDF generation to keep UI responsive."""

    finished = Signal(str)  # PDF file path
    error = Signal(str)
    progress = Signal(int)  # Progress percentage

    def __init__(self, sensors_data, config, output_path):
        super().__init__()
        self.sensors_data = sensors_data
        self.config = config
        self.output_path = output_path

    def run(self):
        """Generate PDF report in background thread."""
        try:
            self.progress.emit(10)
            pdf_path = self.generate_pdf_report()
            self.finished.emit(pdf_path)
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            self.error.emit(str(e))

    def generate_pdf_report(self):
        """Generate comprehensive PDF report with charts and analysis."""
        try:
            logger.info(f"Starting PDF generation to: {self.output_path}")
            
            # Optimized margins for better space utilization while maintaining standardism
            doc = SimpleDocTemplate(
                self.output_path,
                pagesize=A4,
                topMargin=0.6 * inch,  # Reduced from 0.75" - saves 0.15" height
                bottomMargin=0.6 * inch,  # Reduced from 0.75" - saves 0.15" height
                leftMargin=0.6 * inch,  # Reduced from 0.75" - saves 0.15" width
                rightMargin=0.6
                * inch,  # Reduced from 0.75" - saves 0.15" width (total: 0.3" more width, 0.3" more height)
            )

            styles = getSampleStyleSheet()
            logger.debug("Successfully initialized PDF document and styles")

            # Custom styles
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue,
            )

            subtitle_style = ParagraphStyle(
                "CustomSubtitle",
                parent=styles["Heading2"],
                fontSize=18,
                spaceAfter=20,
                alignment=TA_CENTER,
                textColor=colors.darkblue,
                fontName="Helvetica-Bold",
            )

            metadata_style = ParagraphStyle(
                "CustomMetadata",
                parent=styles["Normal"],
                fontSize=11,
                spaceAfter=4,
                alignment=TA_CENTER,
                textColor=colors.grey,
                fontName="Helvetica-Oblique",
            )

            heading_style = ParagraphStyle(
                "CustomHeading",
                parent=styles["Heading2"],
                fontSize=16,
                spaceAfter=12,
                spaceBefore=20,
                textColor=colors.darkblue,
                fontName="Helvetica-Bold",
            )

            subheading_style = ParagraphStyle(
                "CustomSubheading",
                parent=styles["Heading3"],
                fontSize=14,
                spaceAfter=8,
                spaceBefore=12,
                textColor=colors.darkblue,
                fontName="Helvetica-Bold",
            )

            body_style = ParagraphStyle(
                "CustomBody", parent=styles["Normal"], fontSize=11, spaceAfter=6, leading=14
            )

            bold_body_style = ParagraphStyle(
                "CustomBoldBody",
                parent=styles["Normal"],
                fontSize=11,
                spaceAfter=6,
                leading=14,
                fontName="Helvetica-Bold",
            )

            story = []

            # Title Page - always on its own page
            self.progress.emit(20)
            title_content = self._create_title_page(
                title_style, subtitle_style, metadata_style, body_style
            )
            story.extend(title_content)
            story.append(PageBreak())

            # Executive Summary - keep together when possible
            self.progress.emit(30)
            summary_content = self._create_executive_summary(
                heading_style, body_style, bold_body_style
            )
            # Use KeepTogether for executive summary to avoid awkward breaks
            story.append(KeepTogether(summary_content))

            # Strategic page break before comparison table if needed
            story.append(Spacer(1, 0.2 * inch))  # Small buffer space

            # Sensor Comparison Table - intelligent page break handling
            self.progress.emit(40)
            table_content = self._create_comparison_table(heading_style, body_style)
            # Table is already wrapped in KeepTogether within the method
            story.extend(table_content)

            # Charts Section - strategic page break optimization
            self.progress.emit(60)
            if self.config.get("include_bar_charts", False) or self.config.get(
                "include_radar_charts", False
            ):
                # Always start charts on a fresh page for better presentation
                story.append(PageBreak())
                charts_content = self._create_charts_section(
                    heading_style, subheading_style, body_style
                )
                story.extend(charts_content)

            # Detailed Analysis - intelligent section breaks
            self.progress.emit(80)
            analysis_content = self._create_detailed_analysis(
                heading_style, subheading_style, body_style
            )
            # Analysis starts on new page if charts were included, otherwise flows naturally
            if not (
                self.config.get("include_bar_charts", False)
                or self.config.get("include_radar_charts", False)
            ):
                story.append(PageBreak())
            story.extend(analysis_content)

            # Recommendations - keep together when possible
            self.progress.emit(90)
            recommendations_content = self._create_recommendations(
                heading_style, subheading_style, body_style, bold_body_style
            )
            story.extend(recommendations_content)

            # Build PDF
            doc.build(story)
            self.progress.emit(100)
            logger.info("PDF generation completed successfully")

            return self.output_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Detailed error: {str(e)}")
            raise RuntimeError(f"PDF generation failed: {str(e)}")

    def _create_title_page(
        self, title_style, subtitle_style, metadata_style, body_style
    ):
        """Create title page content."""
        story = []

        # Main title
        story.append(Paragraph("Robotics Sensor Comparison Report", title_style))
        story.append(Spacer(1, 0.25 * inch))

        # Subtitle with proper styling
        subtitle = f"Comparative Analysis of {len(self.sensors_data)} Sensors"
        story.append(Paragraph(subtitle, subtitle_style))
        story.append(Spacer(1, 0.3 * inch))

        # Report metadata with italics styling
        report_date = datetime.now().strftime("%B %d, %Y")
        metadata = [
            f"Report Generated: {report_date}",
            f"Sensors Analyzed: {len(self.sensors_data)}",
            f"Export Configuration: {self.config.get('export_type', 'Standard Report')}",
        ]

        for item in metadata:
            story.append(Paragraph(item, metadata_style))

        story.append(Spacer(1, 0.3 * inch))

        # Development disclaimer
        disclaimer_style = ParagraphStyle(
            "DisclaimerStyle",
            parent=metadata_style,
            textColor=colors.red,
            fontName="Helvetica-Oblique",
            fontSize=10,
            spaceAfter=12,
            leftIndent=20,
            rightIndent=20,
        )
        story.append(
            Paragraph(
                "<b>DEVELOPMENT NOTICE:</b> This PDF export feature is under active development. "
                "Report content, formatting, and analysis algorithms are subject to change. "
                "Please verify all technical specifications against manufacturer datasheets.",
                disclaimer_style,
            )
        )
        story.append(Spacer(1, 0.25 * inch))

        # Sensor list with proper bold formatting
        sensor_list_style = ParagraphStyle(
            "SensorListHeader",
            parent=body_style,
            fontName="Helvetica-Bold",
            spaceAfter=8,
        )
        story.append(Paragraph("Sensors in This Report:", sensor_list_style))
        story.append(Spacer(1, 0.15 * inch))

        for i, sensor in enumerate(self.sensors_data, 1):
            manufacturer = sensor.get("manufacturer", "Unknown")
            model = sensor.get("model", "Unknown")
            sensor_line = f"{i}. {manufacturer} {model}"
            story.append(Paragraph(sensor_line, body_style))

        # Add user notes if provided
        user_notes = self.config.get("notes", "").strip()
        if user_notes:
            story.append(Spacer(1, 0.25 * inch))

            notes_header_style = ParagraphStyle(
                "NotesHeader",
                parent=body_style,
                fontName="Helvetica-Bold",
                spaceAfter=8,
            )
            story.append(Paragraph("Additional Notes:", notes_header_style))
            story.append(Spacer(1, 0.1 * inch))

            # Create styled box for user notes
            from reportlab.platypus import Table

            notes_style = ParagraphStyle(
                "NotesContent",
                parent=body_style,
                fontName="Helvetica",
                fontSize=10,
                leftIndent=10,
                rightIndent=10,
                spaceAfter=6,
                textColor=colors.darkblue,
            )

            # Format notes with proper line breaks
            notes_paragraph = Paragraph(user_notes.replace("\n", "<br/>"), notes_style)

            # Create a simple table for styling the notes box
            notes_table = Table([[notes_paragraph]], colWidths=[6.5 * inch])
            notes_table.setStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.aliceblue),
                    ("BORDER", (0, 0), (-1, -1), 1, colors.lightgrey),
                    ("PADDING", (0, 0), (-1, -1), 12),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )

            story.append(notes_table)

        return story

    def _create_executive_summary(self, heading_style, body_style, bold_body_style):
        """Create executive summary section with optimized multi-column layout."""
        story = []

        story.append(Paragraph("Executive Summary", heading_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        story.append(Spacer(1, 12))

        # Generate comprehensive summary statistics
        df = pd.DataFrame(self.sensors_data)

        # Create structured summary data for multi-column layout - fix HTML tags
        key_stats = []
        detailed_insights = []

        # Overview and count
        key_stats.append(f"Sensors Analyzed: {len(self.sensors_data)} robotics sensors")

        # Price analysis with enhanced metrics
        if "price_range" in df.columns:
            prices = df["price_range"].apply(extract_price_avg).dropna()
            if not prices.empty:
                avg_price = prices.mean()
                min_price = prices.min()
                max_price = prices.max()
                median_price = prices.median()
                key_stats.append(f"Price Range: ${min_price:,.0f} - ${max_price:,.0f}")
                key_stats.append(
                    f"Average Price: ${avg_price:,.0f} (Median: ${median_price:,.0f})"
                )

                # Add price distribution insights
                budget_count = len(prices[prices <= 1000])
                premium_count = len(prices[prices > 5000])
                if budget_count > 0:
                    detailed_insights.append(
                        f"Budget-friendly options: {budget_count} sensors under $1,000"
                    )
                if premium_count > 0:
                    detailed_insights.append(
                        f"Modern-grade options: {premium_count} sensors over $5,000"
                    )

        # Performance metrics
        if "frame_rate" in df.columns:
            frame_rates = df["frame_rate"].apply(extract_numeric).dropna()
            if not frame_rates.empty:
                max_fps = frame_rates.max()
                avg_fps = frame_rates.mean()
                key_stats.append(
                    f"Frame Rate: Up to {max_fps} FPS (avg: {avg_fps:.1f})"
                )

                # Performance categories
                high_fps_count = len(frame_rates[frame_rates >= 30])
                if high_fps_count > 0:
                    detailed_insights.append(
                        f"High-performance options: {high_fps_count} sensors ≥30 FPS"
                    )

        # Range capabilities
        if "max_range" in df.columns:
            ranges = df["max_range"].apply(extract_numeric).dropna()
            if not ranges.empty:
                max_range = ranges.max()
                avg_range = ranges.mean()
                key_stats.append(
                    f"Detection Range: Up to {max_range:.1f}m (avg: {avg_range:.1f}m)"
                )

                # Range categories
                long_range_count = len(ranges[ranges >= 10])
                if long_range_count > 0:
                    detailed_insights.append(
                        f"Long-range capabilities: {long_range_count} sensors ≥10m range"
                    )

        # Sensor type diversity
        if "sensor_type" in df.columns:
            sensor_types = df["sensor_type"].value_counts()
            key_stats.append(f"Sensor Types: {len(sensor_types)} different categories")
            detailed_insights.append(
                f"Type distribution: {', '.join([f'{count} {type_}' for type_, count in sensor_types.head(3).items()])}"
            )

        # Manufacturer diversity
        if "manufacturer" in df.columns:
            manufacturers = df["manufacturer"].nunique()
            key_stats.append(f"Manufacturers: {manufacturers} different companies")

        # Create cleaner bullet-point layout instead of complex table
        if key_stats:
            # Add key statistics section with proper headers
            stat_header_style = ParagraphStyle(
                "StatHeader",
                parent=bold_body_style,
                fontName="Helvetica-Bold",
                fontSize=12,
                spaceAfter=8,
                textColor=colors.darkblue,
            )

            story.append(Paragraph("Key Statistics", stat_header_style))

            for stat in key_stats:
                # Create a custom style for stats with bold labels
                parts = stat.split(": ", 1)
                if len(parts) == 2:
                    label, value = parts
                    formatted_stat = f"<b>{label}:</b> {value}"
                else:
                    formatted_stat = f"<b>{stat}</b>"
                story.append(Paragraph(formatted_stat, body_style))

            story.append(Spacer(1, 12))

        if detailed_insights:
            # Add insights section
            insight_header_style = ParagraphStyle(
                "InsightHeader",
                parent=bold_body_style,
                fontName="Helvetica-Bold",
                fontSize=12,
                spaceAfter=8,
                textColor=colors.darkblue,
            )

            story.append(Paragraph("Key Insights", insight_header_style))

            for insight in detailed_insights:
                # Format as bullet points
                story.append(Paragraph(f"• {insight}", body_style))

        # Add analytical conclusion
        story.append(Spacer(1, 12))
        conclusion_text = (
            "This comprehensive analysis provides normalized comparisons across all technical dimensions, "
            "enabling informed sensor selection based on specific application requirements, budget constraints, "
            "and performance criteria."
        )
        story.append(Paragraph(conclusion_text, body_style))
        story.append(Spacer(1, 12))

        # Add data accuracy disclaimer
        disclaimer_header_style = ParagraphStyle(
            "DataDisclaimerHeader",
            parent=bold_body_style,
            fontName="Helvetica-Bold",
            fontSize=10,
            spaceAfter=6,
            textColor=colors.darkred,
        )

        disclaimer_text_style = ParagraphStyle(
            "DataDisclaimerText",
            parent=body_style,
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=colors.grey,
            leftIndent=10,
            rightIndent=10,
            spaceAfter=8,
        )

        story.append(Spacer(1, 8))
        story.append(Paragraph("Data Accuracy Notice", disclaimer_header_style))

        data_disclaimer = (
            "<b>Specifications:</b> All technical specifications are sourced from manufacturer "
            "datasheets and publicly available documentation. While every effort is made to "
            "ensure accuracy, specifications may vary by model revision or region. "
            "<br/><br/>"
            "<b>Pricing:</b> Price information is approximate and based on publicly available "
            "sources at the time of data collection. Actual pricing may vary by distributor, "
            "region, quantity, and market conditions. Always verify current pricing with "
            "official distributors before making purchasing decisions."
        )

        story.append(Paragraph(data_disclaimer, disclaimer_text_style))

        return story

    def _create_comparison_table(self, heading_style, body_style):
        """Create detailed comparison table with optimized space utilization."""
        story = []

        story.append(Paragraph("Sensor Specifications Comparison", heading_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        story.append(Spacer(1, 12))

        # Prepare table data
        df = pd.DataFrame(self.sensors_data)

        # Enhanced column selection with priority ordering for optimal space usage
        priority_columns = [
            ("manufacturer", "text", "Manufacturer"),
            ("model", "text", "Model"),
            ("sensor_type", "text", "Type"),
            ("frame_rate", "numeric", "FPS"),
            ("max_range", "numeric", "Range (m)"),
            ("price_range", "price", "Price"),
            ("resolution", "resolution", "Resolution"),
            ("min_range", "numeric", "Min Range"),
        ]

        # Select available columns and optimize for space
        available_columns = []
        col_types = []
        col_labels = []

        for col, col_type, label in priority_columns:
            if col in df.columns:
                available_columns.append(col)
                col_types.append(col_type)
                col_labels.append(label)

        # Limit to 6 columns max for better readability and space usage
        if len(available_columns) > 6:
            available_columns = available_columns[:6]
            col_types = col_types[:6]
            col_labels = col_labels[:6]

        # Create table headers with optimized labels
        table_data = [col_labels]

        # Add sensor data rows with enhanced formatting
        for _, sensor in df.iterrows():
            row = []
            for i, col in enumerate(available_columns):
                value = sensor.get(col, "N/A")
                col_type = col_types[i]

                if col_type == "numeric":
                    numeric_val = extract_numeric(value)
                    row.append(f"{numeric_val:.1f}" if numeric_val else "N/A")
                elif col_type == "price":
                    price_val = (
                        extract_price_avg(value)
                        if hasattr(value, "__iter__") and not isinstance(value, str)
                        else None
                    )
                    if price_val and price_val > 0:
                        row.append(f"${price_val:,.0f}")
                    else:
                        row.append("N/A")
                elif col_type == "resolution":
                    if isinstance(value, dict) and "rgb" in value:
                        width = value["rgb"].get("width", 0)
                        height = value["rgb"].get("height", 0)
                        row.append(f"{width}×{height}" if width and height else "N/A")
                    else:
                        row.append("N/A")
                else:  # text type
                    # Truncate long text for better table formatting
                    text_val = str(value) if value else "N/A"
                    row.append(
                        text_val[:20] + "..." if len(text_val) > 20 else text_val
                    )
            table_data.append(row)

        # Calculate optimized column widths based on content and page format
        page_format = self.config.get("page_format", "A4 (Portrait)")
        if "Landscape" in page_format:
            if "Letter" in page_format:
                available_width = (
                    9.8 * inch
                )  # Letter landscape with more realistic margins
            else:
                available_width = (
                    10.3 * inch
                )  # A4 landscape with more realistic margins
        else:
            # Portrait mode with optimized space usage
            if "Letter" in page_format:
                available_width = 7.2 * inch  # Letter portrait (optimized from 7.5")
            else:
                available_width = 6.8 * inch  # A4 portrait (optimized from 7")

        # Advanced column width optimization based on content type and length
        if len(available_columns) > 0:
            col_widths = []
            total_weight = 0

            # Assign weights based on column type and content
            for i, col in enumerate(available_columns):
                col_type = col_types[i]
                if col_type == "text" and col in ["manufacturer", "model"]:
                    weight = 2.0  # More space for important text
                elif col_type == "text":
                    weight = 1.5  # Medium space for other text
                elif col_type == "resolution":
                    weight = 1.2  # Medium space for formatted resolution
                elif col_type == "price":
                    weight = 1.0  # Standard space for prices
                else:  # numeric
                    weight = 0.8  # Less space for simple numbers

                col_widths.append(weight)
                total_weight += weight

            # Normalize to available width with minimum column constraints
            normalized_widths = []
            for weight in col_widths:
                base_width = (weight / total_weight) * available_width
                # Ensure minimum readable width (0.8 inch) and maximum reasonable width (2.5 inch)
                normalized_width = max(0.8 * inch, min(2.5 * inch, base_width))
                normalized_widths.append(normalized_width)

            # Final adjustment to fit exact available width
            current_total = sum(normalized_widths)
            col_widths = [
                w * (available_width / current_total) for w in normalized_widths
            ]
        else:
            col_widths = None

        # Create table with enhanced styling for better space utilization
        table = Table(table_data, repeatRows=1, colWidths=col_widths)

        # Enhanced table style with better spacing and readability
        table.setStyle(
            TableStyle(
                [
                    # Header styling
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),  # Optimized header font size
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, 0),
                        6,
                    ),  # Reduced padding for space efficiency
                    ("TOPPADDING", (0, 0), (-1, 0), 6),
                    # Data row styling
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),  # Smaller font for better fit
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 4),  # Compact padding
                    ("TOPPADDING", (0, 1), (-1, -1), 4),
                    # Alignment and formatting
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),  # Center headers
                    (
                        "ALIGN",
                        (0, 1),
                        (-1, -1),
                        "LEFT",
                    ),  # Left align data for readability
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    # Grid and borders
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),  # Thinner grid lines
                    (
                        "LINEBELOW",
                        (0, 0),
                        (-1, 0),
                        1,
                        colors.darkblue,
                    ),  # Stronger header separator
                    # Text wrapping and overflow handling
                    ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
                    # Alternating row colors for better readability
                    ("BACKGROUND", (0, 2), (-1, -1), colors.white),  # Every other row
                    ("BACKGROUND", (0, 3), (-1, -1), colors.beige),
                ]
            )
        )

        # Apply alternating row background for better readability
        for row_idx in range(2, len(table_data), 2):
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, row_idx), (-1, row_idx), colors.white),
                    ]
                )
            )

        story.append(KeepTogether(table))
        story.append(Spacer(1, 16))  # Slightly reduced spacing

        return story

    def _create_charts_section(self, heading_style, subheading_style, body_style):
        """Create charts section with optimized page breaks and layout."""
        story = []

        # Section header (PageBreak handled in main flow)
        story.append(Paragraph("Performance Analysis Charts", heading_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        story.append(Spacer(1, 12))

        charts_added = False

        # Add bar charts if selected
        if self.config.get("include_bar_charts", False):
            bar_chart_paths = self._generate_bar_charts()

            for i, (chart_path, title) in enumerate(bar_chart_paths):
                if os.path.exists(chart_path):
                    # Create chart content as a cohesive unit
                    chart_content = []

                    # Use proper subheading style for chart titles
                    chart_content.append(Paragraph(title, subheading_style))
                    chart_content.append(Spacer(1, 8))

                    # Add chart image with optimized sizing for better space utilization
                    chart_width = min(6.5 * inch, 7 * inch)  # Use more available width
                    chart_height = (
                        chart_width * 0.6
                    )  # Maintain good aspect ratio (3:5 instead of 2:3)
                    img = Image(chart_path, width=chart_width, height=chart_height)
                    chart_content.append(img)
                    chart_content.append(Spacer(1, 8))  # Reduced spacing

                    # Keep chart title and image together
                    story.append(KeepTogether(chart_content))

                    # Add spacing between charts, but not too much
                    if i < len(bar_chart_paths) - 1:
                        story.append(Spacer(1, 8))

                    charts_added = True

        # Add radar charts if selected with intelligent spacing
        radar_charts_enabled = self.config.get("include_radar_charts", False)
        logger.info(f"Radar charts enabled: {radar_charts_enabled}")

        if radar_charts_enabled:
            # Add spacing between bar charts and radar chart if both are present
            if charts_added:
                story.append(Spacer(1, 16))

            logger.info(
                f"Generating radar chart for PDF export with {len(self.sensors_data)} sensors..."
            )
            radar_chart_path = self._generate_radar_chart()
            logger.info(f"Radar chart path returned: {radar_chart_path}")

            if radar_chart_path and os.path.exists(radar_chart_path):
                file_size = os.path.getsize(radar_chart_path)
                logger.info(
                    f"Radar chart generated successfully: {radar_chart_path} (size: {file_size} bytes)"
                )

                # Create radar chart content as cohesive unit
                radar_content = []
                radar_content.append(
                    Paragraph(
                        "Multi-Dimensional Performance Overview", subheading_style
                    )
                )
                radar_content.append(Spacer(1, 8))

                # Add radar chart image with optimized sizing for better space utilization
                try:
                    # Radar charts work well as squares but can be slightly smaller for better page flow
                    radar_size = min(
                        5.5 * inch, 6.5 * inch
                    )  # Slightly smaller for better space usage
                    img = Image(radar_chart_path, width=radar_size, height=radar_size)
                    radar_content.append(img)
                    radar_content.append(Spacer(1, 8))

                    # Keep radar chart title and image together
                    story.append(KeepTogether(radar_content))
                    charts_added = True
                    logger.info("Radar chart successfully added to PDF")
                except Exception as e:
                    logger.error(f"Error adding radar chart image to PDF: {e}")
                    story.append(
                        Paragraph(f"Error displaying radar chart: {str(e)}", body_style)
                    )
                    story.append(Spacer(1, 8))
            else:
                logger.warning(
                    f"Radar chart generation failed or file not found: {radar_chart_path}"
                )
                logger.warning(
                    f"File exists check: {os.path.exists(radar_chart_path) if radar_chart_path else 'Path is None'}"
                )
                story.append(
                    Paragraph(
                        "Radar chart generation failed - please check sensor data compatibility.",
                        body_style,
                    )
                )
                story.append(Spacer(1, 8))

        # Add explanatory text if no charts
        if not charts_added:
            story.append(Paragraph("No charts selected for export.", body_style))
            story.append(Spacer(1, 12))

        return story

    def _generate_bar_charts(self):
        """Generate matplotlib bar charts for PDF inclusion."""
        chart_paths = []
        df = pd.DataFrame(self.sensors_data)

        # Chart 1: Frame Rate Comparison
        if "frame_rate" in df.columns:
            chart_path = self._create_bar_chart(
                df, "frame_rate", "Frame Rate Comparison (FPS)"
            )
            if chart_path:
                chart_paths.append((chart_path, "Frame Rate Performance"))

        # Chart 2: Range Comparison
        if "max_range" in df.columns:
            chart_path = self._create_bar_chart(
                df, "max_range", "Maximum Range Comparison (m)"
            )
            if chart_path:
                chart_paths.append((chart_path, "Maximum Range Capabilities"))

        # Chart 3: Price Comparison
        if "price_range" in df.columns:
            chart_path = self._create_price_chart(df)
            if chart_path:
                chart_paths.append((chart_path, "Price Comparison"))

        return chart_paths

    def _generate_radar_chart(self):
        """Generate Plotly radar chart as static image for PDF inclusion."""
        try:
            # Check for required dependencies first
            try:
                import plotly.graph_objects as go
                from plotly.io import to_image
                import plotly.express as px
            except ImportError as e:
                logger.error(
                    f"Missing required dependencies for radar chart generation: {e}"
                )
                return None

            logger.info(f"Generating radar chart for {len(self.sensors_data)} sensors")
            df = pd.DataFrame(self.sensors_data)

            # Prepare radar chart data using same logic as radar widget
            radar_data = self._prepare_radar_data(df)
            logger.info(f"Radar data prepared for {len(radar_data)} sensors")

            if not radar_data:
                logger.warning(
                    "No radar chart data available - insufficient sensor data"
                )
                return None

            # Create Plotly figure
            fig = go.Figure()

            # Use same color scheme as GUI (Plotly Set1 - unified color scheme)
            colors = px.colors.qualitative.Set1

            logger.info(
                f"Using Plotly Set1 color scheme (unified with GUI): {colors[:len(radar_data)]}"
            )

            for i, (sensor_id, data) in enumerate(radar_data.items()):
                color = colors[i % len(colors)]  # Cycle through Set1 colors
                fig.add_trace(
                    go.Scatterpolar(
                        r=data["values"],
                        theta=data["categories"],
                        fill="toself",
                        name=data["label"],
                        line_color=color,
                        fillcolor=color,
                        opacity=0.6,  # Match GUI opacity
                    )
                )

            # Update layout for better appearance
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickmode="linear",
                        tick0=0,
                        dtick=0.2,
                        showticklabels=True,
                    )
                ),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
                ),
                title=dict(
                    text="Multi-Dimensional Sensor Performance Comparison",
                    x=0.5,
                    font=dict(size=16),
                ),
                width=600,
                height=600,
                margin=dict(t=80, b=80, l=40, r=40),
            )

            # Save as static image with robust error handling
            try:
                export_dpi = self.config.get("export_quality", 300)
                scale_factor = max(1, export_dpi // 150)  # Scale factor based on DPI

                # Generate image bytes using kaleido engine explicitly
                img_bytes = to_image(
                    fig,
                    format="png",
                    width=600,
                    height=600,
                    scale=scale_factor,
                    engine="kaleido",
                )

                # Verify image bytes are valid
                if not img_bytes or len(img_bytes) == 0:
                    logger.error(
                        "Radar chart image generation failed - empty image data"
                    )
                    return None

                # Create temporary file and write image data
                temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                with open(temp_file.name, "wb") as f:
                    f.write(img_bytes)

                # Verify file was written successfully
                if (
                    not os.path.exists(temp_file.name)
                    or os.path.getsize(temp_file.name) == 0
                ):
                    logger.error(
                        "Radar chart file creation failed - file not created or empty"
                    )
                    return None

                file_size = os.path.getsize(temp_file.name)
                logger.info(
                    f"Radar chart generated successfully: {temp_file.name} (size: {file_size} bytes)"
                )

                return temp_file.name

            except Exception as img_error:
                logger.error(f"Failed to generate radar chart image: {img_error}")
                # Check if it's a kaleido-specific error
                if "kaleido" in str(img_error).lower():
                    logger.error(
                        "Kaleido static image export failed. Please ensure kaleido is properly installed: pip install kaleido"
                    )
                return None

        except Exception as e:
            logger.error(f"Error generating radar chart: {e}")
            return None

    def _prepare_radar_data(self, df):
        """Prepare data for radar chart visualization with smart attribute selection for mixed sensor types."""
        # Determine sensor types present in the dataset
        sensor_types = set()
        if "sensor_type" in df.columns:
            sensor_types = set(df["sensor_type"].dropna().unique())

        logger.info(f"Detected sensor types for radar chart: {sensor_types}")

        # Smart attribute selection based on sensor types
        radar_attributes = self._select_radar_attributes(df, sensor_types)

        radar_data = {}

        # Process attributes that might need extraction (same as GUI logic)
        for attr, _ in radar_attributes:
            if attr == "resolution_rgb":
                if "resolution" in df.columns:
                    df = df.copy()
                    df[attr] = df["resolution"].apply(
                        lambda x: (
                            extract_resolution(x, "rgb") if isinstance(x, dict) else 0
                        )
                    )
                else:
                    df[attr] = 0
            elif attr == "price_avg":
                if "price_range" in df.columns:
                    df = df.copy()
                    df[attr] = df["price_range"].apply(extract_price_avg)
                else:
                    df[attr] = 0
            elif attr in df.columns:
                df = df.copy()
                df[attr] = df[attr].apply(extract_numeric)
            else:
                df[attr] = 0

        logger.info(
            f"Processing radar data for {len(df)} sensors with attributes: {[attr for attr, _ in radar_attributes]}"
        )

        # Normalize data for each sensor (same logic as GUI)
        for _, sensor in df.iterrows():
            sensor_id = sensor.get("sensor_id", "Unknown")
            manufacturer = sensor.get("manufacturer", "Unknown")
            model = sensor.get("model", "Unknown")
            values = []
            categories = []

            for attr, direction in radar_attributes:
                if attr in df.columns:
                    # Get value for this sensor
                    value = sensor.get(attr, 0)
                    if pd.isna(value) or value is None:
                        normalized_value = 0
                    else:
                        # Normalize based on min/max in dataset (same as GUI)
                        series = df[attr].apply(
                            lambda x: x if pd.notna(x) and x is not None else 0
                        )
                        min_val = series.min()
                        max_val = series.max()

                        if max_val == min_val:
                            normalized_value = 0.5  # If all values same, put in middle
                        else:
                            if direction == "higher_better":
                                normalized_value = (value - min_val) / (
                                    max_val - min_val
                                )
                            else:  # lower_better
                                normalized_value = (max_val - value) / (
                                    max_val - min_val
                                )

                        # Clamp to 0-1 range
                        normalized_value = max(0, min(1, normalized_value))

                    values.append(normalized_value)
                    categories.append(format_label(attr))

            # Include all sensors with data (no minimum requirement like GUI)
            if values:
                logger.info(
                    f"Sensor {sensor_id}: {len(values)} attributes, values: {[f'{v:.3f}' for v in values[:3]]}"
                )
                radar_data[sensor_id] = {
                    "values": values,
                    "categories": categories,
                    "label": f"{manufacturer} {model}",
                }

        logger.info(
            f"Radar data prepared for {len(radar_data)} sensors using attributes: {[attr for attr, _ in radar_attributes]}"
        )
        return radar_data

    def _select_radar_attributes(self, df, sensor_types):
        """Select appropriate radar chart attributes based on sensor types present."""
        # Define attribute pools by category
        camera_attrs = [
            ("frame_rate", "higher_better"),
            ("resolution_rgb", "higher_better"),
            ("min_range", "lower_better"),
            ("latency", "lower_better"),
        ]

        lidar_attrs = [
            ("max_range", "higher_better"),
            ("angular_resolution", "lower_better"),
            ("points_per_second", "higher_better"),
            ("range_accuracy", "lower_better"),
        ]

        imu_attrs = [
            ("sample_rate", "higher_better"),
            ("noise_density", "lower_better"),
            ("bias_stability", "lower_better"),
            ("operating_range", "higher_better"),
        ]

        # Universal attributes that apply to most sensor types
        universal_attrs = [
            ("price_avg", "lower_better"),
            ("power_consumption", "lower_better"),
            ("weight", "lower_better"),
        ]

        selected_attrs = []

        # Check for camera-like sensors
        camera_types = {
            "RGB Camera",
            "Depth Camera",
            "Infrared Camera",
            "Stereo Camera",
            "Thermal Camera",
            "Time-of-Flight Camera",
            "Structured Light Camera",
        }
        if sensor_types.intersection(camera_types):
            # Add camera attributes that are available
            for attr, direction in camera_attrs:
                if attr == "resolution_rgb" and "resolution" in df.columns:
                    selected_attrs.append((attr, direction))
                elif attr in df.columns and df[attr].notna().any():
                    selected_attrs.append((attr, direction))

        # Check for LiDAR sensors
        lidar_types = {"LiDAR"}
        if sensor_types.intersection(lidar_types):
            for attr, direction in lidar_attrs:
                if attr in df.columns and df[attr].notna().any():
                    selected_attrs.append((attr, direction))

        # Check for IMU sensors (when schema v2 is implemented)
        imu_types = {"IMU", "Gyroscope", "Accelerometer", "Magnetometer"}
        if sensor_types.intersection(imu_types):
            for attr, direction in imu_attrs:
                if attr in df.columns and df[attr].notna().any():
                    selected_attrs.append((attr, direction))

        # Always try to add universal attributes that are available
        for attr, direction in universal_attrs:
            if attr == "price_avg" and "price_range" in df.columns:
                selected_attrs.append((attr, direction))
            elif attr in df.columns and df[attr].notna().any():
                selected_attrs.append((attr, direction))

        # Fallback to basic attributes if no smart selection worked
        if not selected_attrs:
            logger.warning(
                "No sensor-type-specific attributes found, using fallback attributes"
            )
            fallback_attrs = [
                ("frame_rate", "higher_better"),
                ("max_range", "higher_better"),
                ("price_avg", "lower_better"),
                ("latency", "lower_better"),
            ]
            for attr, direction in fallback_attrs:
                if attr == "price_avg" and "price_range" in df.columns:
                    selected_attrs.append((attr, direction))
                elif attr in df.columns and df[attr].notna().any():
                    selected_attrs.append((attr, direction))

        # Ensure we have at least 3 attributes for a meaningful radar chart
        if len(selected_attrs) < 3:
            logger.warning(
                f"Only {len(selected_attrs)} radar attributes selected, radar chart may not be meaningful"
            )

        return selected_attrs[:6]  # Limit to 6 attributes for readability

    def _create_bar_chart(self, df, column, title):
        """Create a bar chart for a specific attribute."""
        try:
            # Extract numeric values
            values = df[column].apply(extract_numeric).dropna()
            labels = [
                f"{row['manufacturer']}\n{row['model']}"
                for _, row in df.iterrows()
                if extract_numeric(row[column]) is not None
            ]

            if values.empty:
                return None

            # Create standard color scheme (unified with GUI)
            standard_colors = [
                "#2E86AB",
                "#A23B72",
                "#F18F01",
                "#C73E1D",
                "#6A994E",
                "#7209B7",
                "#F25C05",
                "#8B5CF6",
                "#059669",
                "#DC2626",
            ]

            # Create gradient colors for better visual appeal
            colors = []
            for i, (_, row) in enumerate(df.iterrows()):
                if extract_numeric(row[column]) is not None:
                    base_color = standard_colors[i % len(standard_colors)]
                    colors.append(base_color)

            # Create chart with enhanced styling
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("#fafafa")

            bars = ax.bar(
                range(len(values)),
                values,
                color=colors,
                alpha=0.8,
                edgecolor="white",
                linewidth=1.5,
            )

            # Enhanced formatting with standard styling
            ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color="#1a202c")
            ax.set_xlabel("Sensors", fontsize=12, fontweight="medium", color="#4a5568")
            ax.set_ylabel(
                column.replace("_", " ").title(),
                fontsize=12,
                fontweight="medium",
                color="#4a5568",
            )

            # Improved x-axis labels
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(
                labels, rotation=45, ha="right", fontsize=10, color="#2d3748"
            )

            # Enhanced grid styling
            ax.grid(axis="y", alpha=0.2, color="#cbd5e0", linestyle="-", linewidth=0.5)
            ax.set_axisbelow(True)

            # Remove top and right spines for cleaner look
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#e2e8f0")
            ax.spines["bottom"].set_color("#e2e8f0")

            # Enhanced value labels on bars with better positioning
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(values) * 0.015,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="medium",
                    color="#2d3748",
                )

            plt.tight_layout()

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            export_dpi = self.config.get("export_quality", 300)
            plt.savefig(temp_file.name, dpi=export_dpi, bbox_inches="tight")
            plt.close()

            return temp_file.name

        except Exception as e:
            logger.error(f"Error creating bar chart for {column}: {e}")
            return None

    def _create_price_chart(self, df):
        """Create a price comparison chart."""
        try:
            prices = df["price_range"].apply(extract_price_avg).dropna()
            labels = [
                f"{row['manufacturer']}\n{row['model']}"
                for _, row in df.iterrows()
                if extract_price_avg(row["price_range"]) is not None
            ]

            if prices.empty:
                return None

            # Create standard color scheme (unified with GUI)
            standard_colors = [
                "#2E86AB",
                "#A23B72",
                "#F18F01",
                "#C73E1D",
                "#6A994E",
                "#7209B7",
                "#F25C05",
                "#8B5CF6",
                "#059669",
                "#DC2626",
            ]

            # Create gradient colors for better visual appeal
            colors = []
            for i, (_, row) in enumerate(df.iterrows()):
                if extract_price_avg(row["price_range"]) is not None:
                    base_color = standard_colors[i % len(standard_colors)]
                    colors.append(base_color)

            # Create chart with enhanced styling
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("#fafafa")

            bars = ax.bar(
                range(len(prices)),
                prices,
                color=colors,
                alpha=0.8,
                edgecolor="white",
                linewidth=1.5,
            )

            # Enhanced formatting with standard styling
            ax.set_title(
                "Price Comparison (USD)",
                fontsize=16,
                fontweight="bold",
                pad=20,
                color="#1a202c",
            )
            ax.set_xlabel("Sensors", fontsize=12, fontweight="medium", color="#4a5568")
            ax.set_ylabel(
                "Price (USD)", fontsize=12, fontweight="medium", color="#4a5568"
            )

            # Improved x-axis labels
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(
                labels, rotation=45, ha="right", fontsize=10, color="#2d3748"
            )

            # Enhanced grid styling
            ax.grid(axis="y", alpha=0.2, color="#cbd5e0", linestyle="-", linewidth=0.5)
            ax.set_axisbelow(True)

            # Remove top and right spines for cleaner look
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#e2e8f0")
            ax.spines["bottom"].set_color("#e2e8f0")

            # Format y-axis as currency with improved styling
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
            ax.tick_params(axis="y", colors="#4a5568")

            # Enhanced value labels on bars with better positioning
            for bar, value in zip(bars, prices):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(prices) * 0.015,
                    f"${value:,.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="medium",
                    color="#2d3748",
                )

            plt.tight_layout()

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            export_dpi = self.config.get("export_quality", 300)
            plt.savefig(temp_file.name, dpi=export_dpi, bbox_inches="tight")
            plt.close()

            return temp_file.name

        except Exception as e:
            logger.error(f"Error creating price chart: {e}")
            return None

    def _create_detailed_analysis(self, heading_style, subheading_style, body_style):
        """Create detailed analysis section with intelligent page break optimization."""
        story = []

        # Section header (not on new page anymore - handled by main flow)
        story.append(Paragraph("Detailed Technical Analysis", heading_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        story.append(Spacer(1, 12))

        df = pd.DataFrame(self.sensors_data)

        # Group sensors for better page flow (analyze 2 sensors per logical group)
        sensors_list = list(df.iterrows())

        for i, (_, sensor) in enumerate(sensors_list):
            manufacturer = sensor.get("manufacturer", "Unknown")
            model = sensor.get("model", "Unknown")

            # Create sensor analysis content as a cohesive unit
            sensor_content = []

            sensor_content.append(
                Paragraph(f"{manufacturer} {model}", subheading_style)
            )

            # Technical specifications in compact format
            specs = []
            if "sensor_type" in sensor:
                specs.append(f"<b>Type:</b> {sensor['sensor_type']}")
            if "frame_rate" in sensor:
                fps = extract_numeric(sensor["frame_rate"])
                if fps:
                    specs.append(f"<b>Frame Rate:</b> {fps} FPS")
            if "max_range" in sensor:
                max_range = extract_numeric(sensor["max_range"])
                if max_range:
                    specs.append(f"<b>Maximum Range:</b> {max_range} m")
            if "price_range" in sensor:
                price = extract_price_avg(sensor["price_range"])
                if price:
                    specs.append(f"<b>Average Price:</b> ${price:,.0f}")

            # Combine specs into fewer paragraphs for better space usage
            if len(specs) <= 2:
                for spec in specs:
                    sensor_content.append(Paragraph(spec, body_style))
            else:
                # Group specs into 2 lines max for compact presentation
                mid_point = len(specs) // 2
                line1 = " | ".join(specs[:mid_point])
                line2 = " | ".join(specs[mid_point:])
                sensor_content.append(Paragraph(line1, body_style))
                sensor_content.append(Paragraph(line2, body_style))

            # Strengths and considerations - more compact format
            strengths = self._analyze_sensor_strengths(sensor)
            if strengths:
                sensor_content.append(Paragraph("<b>Key Strengths:</b>", body_style))
                # Combine strengths into bulleted paragraph for space efficiency
                if len(strengths) <= 3:
                    for strength in strengths:
                        sensor_content.append(Paragraph(f"• {strength}", body_style))
                else:
                    # Compact bulleted list
                    bullet_text = " • ".join(strengths)
                    sensor_content.append(Paragraph(bullet_text, body_style))

            sensor_content.append(Spacer(1, 8))  # Reduced spacing

            # Keep each sensor's analysis together, but allow page breaks between sensors
            story.append(KeepTogether(sensor_content))

            # Add strategic spacing between sensors, but not too much
            if i < len(sensors_list) - 1:  # Don't add space after last sensor
                story.append(Spacer(1, 6))

        return story

    def _analyze_sensor_strengths(self, sensor):
        """Analyze and identify robotics-specific strengths of a sensor based on type."""
        strengths = []
        sensor_type = sensor.get("sensor_type", "Unknown")

        # Sensor-type-specific analysis
        if sensor_type in {
            "RGB Camera",
            "Depth Camera",
            "Stereo Camera",
            "Infrared Camera",
            "Thermal Camera",
            "Time-of-Flight Camera",
            "Structured Light Camera",
        }:
            strengths.extend(self._analyze_camera_strengths(sensor))
        elif sensor_type == "LiDAR":
            strengths.extend(self._analyze_lidar_strengths(sensor))
        elif sensor_type in {"IMU", "Gyroscope", "Accelerometer", "Magnetometer"}:
            strengths.extend(self._analyze_imu_strengths(sensor))
        elif sensor_type in {"Ultrasonic Sensor", "Radar"}:
            strengths.extend(self._analyze_proximity_sensor_strengths(sensor))
        else:
            # Fallback to generic analysis
            strengths.extend(self._analyze_generic_sensor_strengths(sensor))

        # Frame rate analysis with robotics context
        fps = extract_numeric(sensor.get("frame_rate"))
        if fps and fps >= 60:
            strengths.append(
                f"High-speed robotics capable ({fps} FPS - suitable for real-time control)"
            )
        elif fps and fps >= 30:
            strengths.append(
                f"Real-time processing friendly ({fps} FPS - good for navigation)"
            )
        elif fps and fps >= 15:
            strengths.append(
                f"Standard robotics applications ({fps} FPS - adequate for mapping)"
            )

        # Range analysis with application context
        max_range = extract_numeric(sensor.get("max_range"))
        min_range = extract_numeric(sensor.get("min_range"))
        if max_range and min_range:
            if min_range < 0.5 and max_range > 5:
                strengths.append(
                    f"Versatile range ({min_range}-{max_range}m - manipulation to navigation)"
                )
            elif min_range < 0.2:
                strengths.append(
                    f"Close-proximity capable ({min_range}m min - suitable for manipulation)"
                )
            elif max_range > 20:
                strengths.append(
                    f"Long-range detection ({max_range}m - outdoor navigation)"
                )

        # Latency analysis for real-time robotics
        latency = extract_numeric(sensor.get("latency"))
        if latency and latency < 50:
            strengths.append(f"Low latency ({latency}ms - real-time control systems)")
        elif latency and latency < 100:
            strengths.append(
                f"Acceptable latency ({latency}ms - navigation applications)"
            )

        # Power consumption analysis
        power = extract_numeric(sensor.get("power_consumption"))
        if power and power < 5:
            strengths.append(f"Low power consumption ({power}W - mobile robotics)")
        elif power and power < 15:
            strengths.append(f"Moderate power ({power}W - stationary applications)")

        # Price analysis with robotics budget context
        price = extract_price_avg(sensor.get("price_range"))
        if price and price < 500:
            strengths.append("Research/prototype friendly pricing")
        elif price and price < 2000:
            strengths.append("Cost-effective for production systems")
        elif price and price > 5000:
            strengths.append("Industrial-grade specifications and reliability")

        # ROS compatibility with ecosystem context
        ros_compat = sensor.get("ros_compatibility", [])
        if isinstance(ros_compat, list):
            if "ROS2" in ros_compat and "ROS1" in ros_compat:
                strengths.append("Full ROS ecosystem compatibility (ROS1 & ROS2)")
            elif "ROS2" in ros_compat:
                strengths.append("Modern ROS2 support (future-proof)")

        # Platform support for deployment flexibility
        platforms = sensor.get("platform_support", [])
        if isinstance(platforms, list) and len(platforms) >= 2:
            if any("Linux" in str(p) for p in platforms):
                strengths.append("Linux support (standard robotics platform)")

        # Add universal analysis (power, price, ROS compatibility)
        strengths.extend(self._analyze_universal_strengths(sensor))

        # Apply sanitization to all strength descriptions
        return [sanitize_language(strength) for strength in strengths]

    def _analyze_camera_strengths(self, sensor):
        """Analyze strengths specific to camera sensors."""
        strengths = []

        # Frame rate analysis with robotics context
        fps = extract_numeric(sensor.get("frame_rate"))
        if fps and fps >= 60:
            strengths.append(
                f"High-speed robotics capable ({fps} FPS - suitable for real-time control)"
            )
        elif fps and fps >= 30:
            strengths.append(
                f"Real-time processing friendly ({fps} FPS - good for navigation)"
            )
        elif fps and fps >= 15:
            strengths.append(
                f"Standard robotics applications ({fps} FPS - adequate for mapping)"
            )

        # Range analysis for depth cameras
        max_range = extract_numeric(sensor.get("max_range"))
        min_range = extract_numeric(sensor.get("min_range"))
        if max_range and min_range:
            if min_range < 0.5 and max_range > 5:
                strengths.append(
                    f"Versatile range ({min_range}-{max_range}m - manipulation to navigation)"
                )
            elif min_range < 0.2:
                strengths.append(
                    f"Close-proximity capable ({min_range}m min - suitable for manipulation)"
                )
            elif max_range > 20:
                strengths.append(
                    f"Long-range detection ({max_range}m - outdoor navigation)"
                )

        # Resolution analysis
        resolution = sensor.get("resolution", {})
        if isinstance(resolution, dict) and "rgb" in resolution:
            rgb_res = resolution["rgb"]
            if isinstance(rgb_res, dict):
                width = rgb_res.get("width", 0)
                height = rgb_res.get("height", 0)
                if width >= 1920 and height >= 1080:
                    strengths.append(
                        f"High-resolution imaging ({width}x{height} - detailed perception)"
                    )

        # Latency analysis for real-time robotics
        latency = extract_numeric(sensor.get("latency"))
        if latency and latency < 50:
            strengths.append(f"Low latency ({latency}ms - real-time control systems)")
        elif latency and latency < 100:
            strengths.append(
                f"Acceptable latency ({latency}ms - navigation applications)"
            )

        return strengths

    def _analyze_lidar_strengths(self, sensor):
        """Analyze strengths specific to LiDAR sensors."""
        strengths = []

        # Range performance
        max_range = extract_numeric(sensor.get("max_range"))
        if max_range:
            if max_range > 100:
                strengths.append(
                    f"Long-range perception ({max_range}m - outdoor autonomous vehicles)"
                )
            elif max_range > 30:
                strengths.append(
                    f"Medium-range detection ({max_range}m - navigation and mapping)"
                )
            else:
                strengths.append(
                    f"Short-range precision ({max_range}m - indoor robotics)"
                )

        # Angular resolution
        angular_res = sensor.get("angular_resolution", {})
        if isinstance(angular_res, dict) and "horizontal" in angular_res:
            h_res = extract_numeric(angular_res["horizontal"])
            if h_res and h_res < 0.1:
                strengths.append(
                    f"High angular precision ({h_res}° - detailed environment mapping)"
                )

        # Channels for multi-channel LiDAR
        channels = extract_numeric(sensor.get("channels"))
        if channels:
            if channels >= 64:
                strengths.append(
                    f"High-density scanning ({channels} channels - automotive grade)"
                )
            elif channels >= 16:
                strengths.append(
                    f"Multi-layer scanning ({channels} channels - 3D mapping)"
                )

        # Points per second
        pps = sensor.get("points_per_second", {})
        if isinstance(pps, dict):
            single_return = extract_numeric(pps.get("single_return", 0))
            if single_return and single_return > 1000000:
                strengths.append(
                    f"High point density ({single_return/1000000:.1f}M pts/sec - detailed reconstruction)"
                )

        return strengths

    def _analyze_imu_strengths(self, sensor):
        """Analyze strengths specific to IMU sensors."""
        strengths = []

        # Sample rate analysis
        sample_rate = extract_numeric(sensor.get("sample_rate"))
        if sample_rate:
            if sample_rate >= 1000:
                strengths.append(
                    f"High-frequency sampling ({sample_rate}Hz - precision control)"
                )
            elif sample_rate >= 100:
                strengths.append(
                    f"Standard sampling rate ({sample_rate}Hz - navigation systems)"
                )

        # Noise characteristics
        noise_density = extract_numeric(sensor.get("noise_density"))
        if noise_density and noise_density < 0.1:
            strengths.append(
                f"Low noise characteristics ({noise_density} - precise orientation)"
            )

        # Operating range
        operating_range = extract_numeric(sensor.get("operating_range"))
        if operating_range:
            if operating_range >= 2000:
                strengths.append(
                    f"Wide operating range (±{operating_range}°/s - dynamic applications)"
                )

        return strengths

    def _analyze_proximity_sensor_strengths(self, sensor):
        """Analyze strengths specific to ultrasonic and radar sensors."""
        strengths = []

        # Range analysis
        max_range = extract_numeric(sensor.get("max_range"))
        min_range = extract_numeric(sensor.get("min_range"))
        if max_range and min_range:
            if max_range > 5:
                strengths.append(
                    f"Medium-range detection ({max_range}m - obstacle avoidance)"
                )
            if min_range < 0.05:
                strengths.append(
                    f"Close-proximity detection ({min_range}m - precise positioning)"
                )

        # Response time
        response_time = extract_numeric(sensor.get("response_time"))
        if response_time and response_time < 100:
            strengths.append(
                f"Fast response time ({response_time}ms - real-time feedback)"
            )

        return strengths

    def _analyze_generic_sensor_strengths(self, sensor):
        """Generic analysis for unknown sensor types."""
        strengths = []

        # Basic range analysis if available
        max_range = extract_numeric(sensor.get("max_range"))
        if max_range:
            strengths.append(f"Detection range: {max_range}m")

        # Basic sampling/frame rate if available
        fps = extract_numeric(sensor.get("frame_rate"))
        sample_rate = extract_numeric(sensor.get("sample_rate"))
        if fps:
            strengths.append(f"Update rate: {fps} FPS")
        elif sample_rate:
            strengths.append(f"Sample rate: {sample_rate} Hz")

        return strengths

    def _analyze_universal_strengths(self, sensor):
        """Analyze universal strengths applicable to all sensor types."""
        strengths = []

        # Power consumption analysis
        power = extract_numeric(sensor.get("power_consumption"))
        if power and power < 5:
            strengths.append(f"Low power consumption ({power}W - mobile robotics)")
        elif power and power < 15:
            strengths.append(f"Moderate power ({power}W - stationary applications)")

        # Price analysis with robotics budget context
        price = extract_price_avg(sensor.get("price_range"))
        if price and price < 500:
            strengths.append("Research/prototype friendly pricing")
        elif price and price < 2000:
            strengths.append("Cost-effective for production systems")
        elif price and price > 5000:
            strengths.append("Industrial-grade specifications and reliability")

        # ROS compatibility with ecosystem context
        ros_compat = sensor.get("ros_compatibility", [])
        if isinstance(ros_compat, list):
            if "ROS2" in ros_compat and "ROS1" in ros_compat:
                strengths.append("Full ROS ecosystem compatibility (ROS1 & ROS2)")
            elif "ROS2" in ros_compat:
                strengths.append("Modern ROS2 support (future-proof)")

        # Environmental rating
        env_rating = sensor.get("environmental_rating")
        if env_rating and env_rating != "None":
            strengths.append(
                f"Environmental protection ({env_rating} - outdoor applications)"
            )

        # Weight consideration for mobile applications
        weight = extract_numeric(sensor.get("weight"))
        if weight and weight < 0.1:  # Less than 100g
            strengths.append(f"Lightweight design ({weight}kg - UAV compatible)")
        elif weight and weight < 0.5:  # Less than 500g
            strengths.append(f"Moderate weight ({weight}kg - mobile robot suitable)")

        return strengths

    def _analyze_robotics_use_cases(self, df):
        """Analyze sensor suitability for common robotics applications."""
        use_case_analysis = []

        # Navigation and SLAM applications
        nav_suitable = []
        for idx, sensor in df.iterrows():
            fps = extract_numeric(sensor.get("frame_rate"))
            max_range = extract_numeric(sensor.get("max_range"))
            latency = extract_numeric(sensor.get("latency"))

            score = 0
            reasons = []

            if fps and fps >= 20:
                score += 2
                reasons.append(f"adequate frame rate ({fps} FPS)")
            if max_range and max_range >= 5:
                score += 2
                reasons.append(f"sufficient range ({max_range}m)")
            if latency and latency <= 100:
                score += 1
                reasons.append(f"acceptable latency ({latency}ms)")

            if score >= 3:
                nav_suitable.append(
                    {
                        "sensor": f"{sensor['manufacturer']} {sensor['model']}",
                        "reasons": reasons,
                        "score": score,
                    }
                )

        if nav_suitable:
            top_nav = sorted(nav_suitable, key=lambda x: x["score"], reverse=True)[:2]
            use_case_analysis.append(
                f"Navigation & SLAM: {top_nav[0]['sensor']} shows strong suitability "
                f"({', '.join(top_nav[0]['reasons'])})."
            )

        # Manipulation and close-range tasks
        manip_suitable = []
        for idx, sensor in df.iterrows():
            min_range = extract_numeric(sensor.get("min_range"))
            fps = extract_numeric(sensor.get("frame_rate"))
            latency = extract_numeric(sensor.get("latency"))

            score = 0
            reasons = []

            if min_range and min_range <= 0.5:
                score += 3
                reasons.append(f"close-range capability ({min_range}m min)")
            if fps and fps >= 30:
                score += 2
                reasons.append(f"real-time feedback ({fps} FPS)")
            if latency and latency <= 50:
                score += 2
                reasons.append(f"low latency ({latency}ms)")

            if score >= 4:
                manip_suitable.append(
                    {
                        "sensor": f"{sensor['manufacturer']} {sensor['model']}",
                        "reasons": reasons,
                        "score": score,
                    }
                )

        if manip_suitable:
            top_manip = sorted(manip_suitable, key=lambda x: x["score"], reverse=True)[
                0
            ]
            analysis_text = (
                f"Manipulation Tasks: {top_manip['sensor']} demonstrates excellent "
                f"close-range capabilities ({', '.join(top_manip['reasons'])})."
            )
            use_case_analysis.append(sanitize_language(analysis_text))

        # Mobile robotics (battery-powered systems)
        mobile_suitable = []
        for idx, sensor in df.iterrows():
            power = extract_numeric(sensor.get("power_consumption"))
            price = extract_price_avg(sensor.get("price_range"))

            score = 0
            reasons = []

            if power and power <= 10:
                score += 2
                reasons.append(f"low power consumption ({power}W)")
            if price and price <= 2000:
                score += 1
                reasons.append("mobile-friendly pricing")
            # Note: Most robotics sensors support Linux, so this is not a distinguishing factor

            if score >= 1:  # Adjusted threshold since Linux compatibility removed
                mobile_suitable.append(
                    {
                        "sensor": f"{sensor['manufacturer']} {sensor['model']}",
                        "reasons": reasons,
                        "score": score,
                    }
                )

        if mobile_suitable:
            top_mobile = sorted(
                mobile_suitable, key=lambda x: x["score"], reverse=True
            )[0]
            use_case_analysis.append(
                f"Mobile Robotics: {top_mobile['sensor']} offers mobile-optimized features "
                f"({', '.join(top_mobile['reasons'])})."
            )

        return use_case_analysis

    def _generate_technical_trade_offs(self, df):
        """Generate analysis of key technical trade-offs for engineering decisions."""
        trade_offs = []

        # Performance vs Cost analysis
        perf_cost_data = []
        for idx, sensor in df.iterrows():
            fps = extract_numeric(sensor.get("frame_rate", 0)) or 0
            max_range = extract_numeric(sensor.get("max_range", 0)) or 0
            price = extract_price_avg(sensor.get("price_range")) or 0
            latency = extract_numeric(sensor.get("latency", 100)) or 100

            # Calculate performance score (higher is better)
            perf_score = (fps * 0.3) + (max_range * 0.3) + ((200 - latency) / 200 * 0.4)

            if price > 0:
                value_ratio = perf_score / (price / 1000)  # Performance per $1k
                perf_cost_data.append(
                    {
                        "sensor": f"{sensor['manufacturer']} {sensor['model']}",
                        "performance_score": perf_score,
                        "price": price,
                        "value_ratio": value_ratio,
                    }
                )

        if perf_cost_data:
            best_value = max(perf_cost_data, key=lambda x: x["value_ratio"])
            highest_perf = max(perf_cost_data, key=lambda x: x["performance_score"])

            # Handle case where same sensor is both best value and highest performance
            if best_value["sensor"] == highest_perf["sensor"]:
                trade_off_text = (
                    f"Performance vs Cost Analysis: {best_value['sensor']} provides "
                    f"both strong performance and good value at ${best_value['price']:,.0f}."
                )
            else:
                trade_off_text = (
                    f"Performance vs Cost Trade-off: {best_value['sensor']} offers good "
                    f"performance-to-cost ratio (${best_value['price']:,.0f}), while "
                    f"{highest_perf['sensor']} provides peak performance capabilities "
                    f"(${highest_perf['price']:,.0f})."
                )
            trade_offs.append(sanitize_language(trade_off_text))

        # Range vs Power trade-off
        range_power_data = []
        for idx, sensor in df.iterrows():
            max_range = extract_numeric(sensor.get("max_range", 0)) or 0
            power = extract_numeric(sensor.get("power_consumption", 0)) or 0

            if max_range > 0 and power > 0:
                efficiency = max_range / power  # Meters per Watt
                range_power_data.append(
                    {
                        "sensor": f"{sensor['manufacturer']} {sensor['model']}",
                        "range": max_range,
                        "power": power,
                        "efficiency": efficiency,
                    }
                )

        if range_power_data:
            most_efficient = max(range_power_data, key=lambda x: x["efficiency"])
            trade_offs.append(
                f"Range vs Power Efficiency: {most_efficient['sensor']} achieves "
                f"{most_efficient['efficiency']:.1f} meters per watt "
                f"({most_efficient['range']}m range, {most_efficient['power']}W consumption)."
            )

        return trade_offs

    def _create_recommendations(
        self, heading_style, subheading_style, body_style, bold_body_style
    ):
        """Create decision support section with factual analysis and optimized layout."""
        story = []

        # Add strategic spacing before recommendations section
        story.append(Spacer(1, 0.3 * inch))

        story.append(Paragraph("Decision Support Information", heading_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        story.append(Spacer(1, 12))

        df = pd.DataFrame(self.sensors_data)

        # Robotics Use Case Analysis
        story.append(Paragraph("Robotics Application Analysis", subheading_style))
        story.append(Spacer(1, 6))

        use_case_analysis = self._analyze_robotics_use_cases(df)
        for analysis in use_case_analysis:
            # Clean up HTML tags in analysis content
            cleaned_analysis = analysis.replace("<b>", "").replace("</b>", "")
            story.append(Paragraph(f"• {cleaned_analysis}", body_style))
            story.append(Spacer(1, 4))

        if not use_case_analysis:
            story.append(
                Paragraph(
                    "Analysis pending - insufficient sensor data for use case comparison.",
                    body_style,
                )
            )
            story.append(Spacer(1, 8))
        elif use_case_analysis:
            story.append(Spacer(1, 8))

        # Technical Trade-offs Analysis
        story.append(Paragraph("Engineering Trade-offs", subheading_style))
        story.append(Spacer(1, 6))

        trade_offs = self._generate_technical_trade_offs(df)
        for trade_off in trade_offs:
            # Clean up HTML tags in trade-off content
            cleaned_trade_off = trade_off.replace("<b>", "").replace("</b>", "")
            story.append(Paragraph(f"• {cleaned_trade_off}", body_style))
            story.append(Spacer(1, 4))

        if not trade_offs:
            story.append(
                Paragraph(
                    "Analysis pending - insufficient sensor data for trade-off comparison.",
                    body_style,
                )
            )
            story.append(Spacer(1, 8))
        elif trade_offs:
            story.append(Spacer(1, 8))

        # Enhanced Performance Analysis
        observations = []

        # More sophisticated price analysis
        if "price_range" in df.columns:
            prices = df["price_range"].apply(extract_price_avg).dropna()
            if not prices.empty:
                min_price_idx = prices.idxmin()
                max_price_idx = prices.idxmax()
                avg_price = prices.mean()
                lowest_cost = df.iloc[min_price_idx]
                highest_cost = df.iloc[max_price_idx]

                observations.append(
                    f"Budget Analysis: Options range from ${extract_price_avg(lowest_cost['price_range']):,.0f} "
                    f"({lowest_cost['manufacturer']} {lowest_cost['model']}) to "
                    f"${extract_price_avg(highest_cost['price_range']):,.0f} "
                    f"({highest_cost['manufacturer']} {highest_cost['model']}), "
                    f"with an average of ${avg_price:,.0f} across all sensors."
                )

        # Performance clustering analysis
        if "frame_rate" in df.columns:
            fps_values = df["frame_rate"].apply(extract_numeric).dropna()
            if not fps_values.empty:
                high_perf = df[df["frame_rate"].apply(extract_numeric) >= 60]

                if not high_perf.empty:
                    high_perf_names = [
                        f"{s['manufacturer']} {s['model']}"
                        for _, s in high_perf.iterrows()
                    ]
                    observations.append(
                        f"High-Performance Options: {', '.join(high_perf_names)} "
                        f"offer 60+ FPS capabilities suitable for real-time control applications."
                    )

        # Robotics-specific application considerations
        observations.extend(
            [
                "Autonomous Navigation: Consider sensors with 5+ meter range, 20+ FPS, and sub-100ms latency for reliable SLAM and obstacle avoidance.",
                "Robotic Manipulation: Sub-meter minimum range and low latency (sub-50ms) enable precise object detection and grasping tasks.",
                "Mobile Robotics: Power consumption under 10W and Linux compatibility are critical for battery-powered autonomous systems.",
                "Industrial Automation: Higher-cost sensors often provide enhanced reliability and accuracy required for production environments.",
                "Research & Development: Budget-friendly options with ROS2 support enable rapid prototyping and algorithm development.",
            ]
        )

        # Add comprehensive analysis section with better organization
        if observations:
            story.append(Paragraph("Application Guidelines", subheading_style))
            story.append(Spacer(1, 6))

            for obs in observations:
                # Format observations with bold labels
                parts = obs.split(": ", 1)
                if len(parts) == 2:
                    label, description = parts
                    formatted_obs = f"<b>{label}:</b> {description}"
                else:
                    formatted_obs = f"• {obs}"
                story.append(Paragraph(formatted_obs, body_style))
                story.append(Spacer(1, 4))

            story.append(Spacer(1, 8))

        # Add neutral disclaimer as a cohesive block
        disclaimer_content = []
        disclaimer_content.append(Spacer(1, 12))

        # Create italic style for disclaimer
        disclaimer_style = ParagraphStyle(
            "DisclaimerStyle",
            parent=body_style,
            fontName="Helvetica-Oblique",
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_LEFT,
        )

        disclaimer_content.append(
            Paragraph(
                "Engineering Decision Note: This analysis provides technical comparisons and robotics-focused insights based on documented specifications. "
                "Final sensor selection should integrate specific application requirements, environmental constraints, system integration complexity, "
                "and project timeline considerations. Always validate performance characteristics through testing in your target deployment environment.",
                disclaimer_style,
            )
        )
        disclaimer_content.append(Spacer(1, 8))

        # Keep disclaimer together
        story.append(KeepTogether(disclaimer_content))

        return story


class PDFExportWidget(QWidget):
    """Widget for configuring and generating PDF export reports."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sensors_data = []
        self.pdf_thread = None

        # Get theme manager
        self.theme_manager = get_theme_manager()

        self.setup_ui()

    def setup_ui(self):
        """Initialize the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)

        # Create scroll area for better accessibility on smaller screens
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create content widget that will be scrollable
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # Export configuration
        config_group = QGroupBox("PDF Export Configuration")
        config_group.setFont(create_styled_font("h4", "medium"))  # Better hierarchy
        config_layout = QVBoxLayout(config_group)

        # Report title
        title_layout = QHBoxLayout()
        title_label = QLabel("Report Title:")
        title_label.setFont(
            create_styled_font("field_label", "medium")
        )  # Proper label size
        title_layout.addWidget(title_label)
        self.title_input = QLineEdit("Sensor Comparison Report")
        self.title_input.setFont(create_styled_font("body"))
        title_layout.addWidget(self.title_input)
        config_layout.addLayout(title_layout)

        # Content options
        content_layout = QVBoxLayout()

        # Chart selection options
        chart_group = QGroupBox("Chart Options")
        chart_group.setFont(create_styled_font("h5", "medium"))  # Better hierarchy
        chart_layout = QVBoxLayout(chart_group)

        chart_type_layout = QHBoxLayout()
        self.include_bar_charts = QCheckBox("Include Bar Charts")
        self.include_bar_charts.setFont(create_styled_font("body"))  # Consistent font
        self.include_bar_charts.setChecked(True)
        chart_type_layout.addWidget(self.include_bar_charts)

        self.include_radar_charts = QCheckBox("Include Radar Charts")
        self.include_radar_charts.setFont(create_styled_font("body"))  # Consistent font
        self.include_radar_charts.setChecked(True)
        chart_type_layout.addWidget(self.include_radar_charts)

        chart_layout.addLayout(chart_type_layout)

        # Chart convenience options
        chart_convenience_layout = QHBoxLayout()
        self.charts_all_button = QPushButton("Select All Charts")
        self.charts_all_button.setFont(
            create_styled_font("button")
        )  # Consistent button font
        self.charts_all_button.setToolTip(
            "Include all available chart types in the PDF report for comprehensive analysis"
        )
        self.charts_all_button.clicked.connect(self.select_all_charts)
        chart_convenience_layout.addWidget(self.charts_all_button)

        self.charts_none_button = QPushButton("Select No Charts")
        self.charts_none_button.setFont(
            create_styled_font("button")
        )  # Consistent button font
        self.charts_none_button.setToolTip(
            "Generate a text-only report without any charts or visualizations"
        )
        self.charts_none_button.clicked.connect(self.select_no_charts)
        chart_convenience_layout.addWidget(self.charts_none_button)

        chart_layout.addLayout(chart_convenience_layout)
        content_layout.addWidget(chart_group)

        # Other content options
        other_options_layout = QHBoxLayout()

        self.include_analysis = QCheckBox("Include Technical Analysis")
        self.include_analysis.setChecked(True)
        other_options_layout.addWidget(self.include_analysis)

        self.include_recommendations = QCheckBox("Include Decision Support")
        self.include_recommendations.setChecked(True)
        other_options_layout.addWidget(self.include_recommendations)

        content_layout.addLayout(other_options_layout)

        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_group.setFont(create_styled_font("h5", "medium"))  # Better hierarchy
        advanced_layout = QVBoxLayout(advanced_group)

        # Export quality
        quality_layout = QHBoxLayout()
        quality_label = QLabel("Export Quality:")
        quality_label.setFont(
            create_styled_font("field_label", "medium")
        )  # Proper label size
        quality_layout.addWidget(quality_label)
        self.quality_combo = QComboBox()
        self.quality_combo.setFont(create_styled_font("body"))  # Consistent font
        self.quality_combo.addItems(
            ["Standard (300 DPI)", "High (600 DPI)", "Print Ready (1200 DPI)"]
        )
        self.quality_combo.setCurrentIndex(0)
        quality_layout.addWidget(self.quality_combo)
        quality_layout.addStretch()
        advanced_layout.addLayout(quality_layout)

        # Page format
        format_layout = QHBoxLayout()
        format_label = QLabel("Page Format:")
        format_label.setFont(
            create_styled_font("field_label", "medium")
        )  # Proper label size
        format_layout.addWidget(format_label)
        self.format_combo = QComboBox()
        self.format_combo.setFont(create_styled_font("body"))  # Consistent font
        self.format_combo.addItems(
            [
                "A4 (Portrait)",
                "A4 (Landscape)",
                "Letter (Portrait)",
                "Letter (Landscape)",
            ]
        )
        self.format_combo.setCurrentIndex(0)
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        advanced_layout.addLayout(format_layout)

        content_layout.addWidget(advanced_group)
        config_layout.addLayout(content_layout)

        # Add proper spacing before notes section
        config_layout.addSpacing(15)

        # Custom notes
        notes_label = QLabel("Additional Notes:")
        notes_label.setFont(
            create_styled_font("section_header", "medium")
        )  # Better visibility and hierarchy
        config_layout.addWidget(notes_label)

        # Reduced spacing after the label for better visual hierarchy
        config_layout.addSpacing(5)

        self.notes_input = QTextEdit()
        self.notes_input.setMaximumHeight(80)
        self.notes_input.setFont(create_styled_font("body"))
        self.notes_input.setPlaceholderText(
            "Enter any additional notes or context for the report..."
        )
        config_layout.addWidget(self.notes_input)

        layout.addWidget(config_group)

        # Export controls
        controls_group = QGroupBox("Export Controls")
        controls_layout = QHBoxLayout(controls_group)

        # File selection
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("Select output file path...")
        self.file_path_input.setFont(create_styled_font("body"))
        controls_layout.addWidget(self.file_path_input)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.setFont(create_styled_font("body"))
        self.browse_button.setToolTip("Choose where to save the PDF report file")
        self.browse_button.clicked.connect(self.browse_output_file)
        controls_layout.addWidget(self.browse_button)

        layout.addWidget(controls_group)

        # Generation controls
        gen_layout = QHBoxLayout()

        self.generate_button = QPushButton("Generate PDF Report")
        self.generate_button.setFont(
            create_styled_font("button", "medium")
        )  # Better button font
        self.generate_button.setToolTip(
            "Generate a comprehensive PDF report with selected sensors, charts, and analysis"
        )
        self.generate_button.clicked.connect(self.generate_pdf)
        self.generate_button.setEnabled(False)
        gen_layout.addWidget(self.generate_button)

        gen_layout.addStretch()

        self.open_button = QPushButton("Open Report")
        self.open_button.setFont(create_styled_font("button"))  # Consistent button font
        self.open_button.setToolTip(
            "Open the generated PDF report in your default PDF viewer"
        )
        self.open_button.clicked.connect(self.open_generated_report)
        self.open_button.setEnabled(False)
        gen_layout.addWidget(self.open_button)

        layout.addLayout(gen_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel(
            "Select sensors and configure export options to generate PDF report."
        )
        self.status_label.setFont(create_styled_font("body"))
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Development disclaimer
        disclaimer_frame = QFrame()
        disclaimer_frame.setFrameStyle(QFrame.Box)
        disclaimer_frame.setLineWidth(1)
        disclaimer_layout = QHBoxLayout(disclaimer_frame)
        disclaimer_layout.setContentsMargins(8, 6, 8, 6)

        # Warning icon (using Unicode symbol)
        icon_label = QLabel("⚠️")
        icon_label.setFont(create_styled_font("body", "bold"))
        disclaimer_layout.addWidget(icon_label)

        disclaimer_text = QLabel(
            "<b>Development Notice:</b> This PDF export feature is under active development. "
            "Report content and analysis algorithms are subject to change. "
            "Please verify technical specifications against manufacturer datasheets."
        )
        disclaimer_text.setFont(create_styled_font("body"))
        disclaimer_text.setWordWrap(True)
        disclaimer_text.setStyleSheet("color: #d69e2e; font-style: italic;")
        disclaimer_layout.addWidget(disclaimer_text, 1)

        layout.addWidget(disclaimer_frame)
        layout.addSpacing(6)

        # Data accuracy disclaimer
        data_disclaimer_frame = QFrame()
        data_disclaimer_frame.setFrameStyle(QFrame.Box)
        data_disclaimer_frame.setLineWidth(1)
        data_disclaimer_layout = QVBoxLayout(data_disclaimer_frame)
        data_disclaimer_layout.setContentsMargins(8, 6, 8, 6)

        data_header_layout = QHBoxLayout()
        data_icon_label = QLabel("📈")  # Chart icon
        data_icon_label.setFont(create_styled_font("body", "bold"))
        data_header_layout.addWidget(data_icon_label)

        data_title_label = QLabel("<b>Data Accuracy Notice</b>")
        data_title_label.setFont(create_styled_font("body", "bold"))
        data_title_label.setStyleSheet("color: #c53030; margin-left: 5px;")
        data_header_layout.addWidget(data_title_label)
        data_header_layout.addStretch()

        data_disclaimer_layout.addLayout(data_header_layout)

        data_disclaimer_text = QLabel(
            "<b>Specifications:</b> Technical data sourced from manufacturer datasheets. "
            "Specifications may vary by model revision or region.<br/>"
            "<b>Pricing:</b> Price information is approximate and may vary by distributor, "
            "region, and market conditions. Always verify current pricing with official sources."
        )
        data_disclaimer_text.setFont(create_styled_font("body"))
        data_disclaimer_text.setWordWrap(True)
        data_disclaimer_text.setStyleSheet(
            "color: #4a5568; font-style: italic; margin-top: 5px;"
        )
        data_disclaimer_layout.addWidget(data_disclaimer_text)

        layout.addWidget(data_disclaimer_frame)
        layout.addSpacing(6)

        # Add stretch to push content to top
        layout.addStretch()

        # Set the content widget in the scroll area
        scroll_area.setWidget(content_widget)

        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)

        # Apply theme-aware styling
        self.apply_theme()

        # Connect to theme changes
        self.theme_manager.theme_changed.connect(self.apply_theme)

    def apply_theme(self):
        """Apply current theme styling to the widget."""
        # Get basic styling
        colors = self.theme_manager.get_stylesheet_colors()
        button_style = self.theme_manager.create_button_stylesheet("primary")

        # Create standardized widget styling with enhanced GroupBox support
        widget_style = f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {colors['border']};
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 15px;
                color: {colors['text_primary']};
                background-color: {colors['surface']};
                font-size: 12px;
            }}
            QGroupBox::title {{
                background-color: {colors['surface']};
                color: {colors['text_primary']};
                font-weight: bold;
                font-size: 13px;
                border: 1px solid {colors['border']};
                border-radius: 3px;
                padding: 0 8px 0 8px;
                left: 10px;
                subcontrol-origin: margin;
                subcontrol-position: top left;
            }}
            QLabel {{
                color: {colors['text_primary']};
                background-color: transparent;
            }}
            QCheckBox {{
                color: {colors['text_primary']};
                background-color: transparent;
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid {colors['border']};
                border-radius: 3px;
                background-color: {colors['surface']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {colors['primary']};
                border-color: {colors['primary']};
            }}
            QCheckBox::indicator:hover {{
                border-color: {colors['border_focus']};
            }}
            QComboBox {{
                color: {colors['text_primary']};
                background-color: {colors['surface']};
                border: 2px solid {colors['border']};
                border-radius: 3px;
                padding: 4px 8px;
                min-width: 100px;
            }}
            QComboBox:hover {{
                border-color: {colors['border_focus']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {colors['text_primary']};
                margin-right: 5px;
            }}
            QLineEdit {{
                color: {colors['text_primary']};
                background-color: {colors['surface']};
                border: 2px solid {colors['border']};
                border-radius: 3px;
                padding: 4px 8px;
            }}
            QLineEdit:focus {{
                border-color: {colors['border_focus']};
            }}
            QTextEdit {{
                color: {colors['text_primary']};
                background-color: {colors['surface']};
                border: 2px solid {colors['border']};
                border-radius: 3px;
                padding: 4px 8px;
            }}
            QTextEdit:focus {{
                border-color: {colors['border_focus']};
            }}
        """

        # Combine stylesheets
        combined_style = widget_style + "\n" + button_style
        self.setStyleSheet(combined_style)

    def update_sensors_data(self, sensors_data: List[Dict[str, Any]]):
        """Update sensor data for PDF export."""
        self.sensors_data = sensors_data.copy()

        if len(sensors_data) >= 2:
            self.generate_button.setEnabled(True)
            self.status_label.setText(
                f"Ready to generate PDF report for {len(sensors_data)} sensors."
            )
        else:
            self.generate_button.setEnabled(False)
            self.status_label.setText(
                "Select at least 2 sensors to generate PDF report."
            )

        logger.info(f"PDF export widget updated with {len(sensors_data)} sensors")

    def browse_output_file(self):
        """Browse for output file location."""
        default_name = (
            f"sensor_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", default_name, "PDF Files (*.pdf);;All Files (*)"
        )

        if file_path:
            self.file_path_input.setText(file_path)

    def generate_pdf(self):
        """Generate PDF report."""
        if not self.sensors_data or len(self.sensors_data) < 2:
            QMessageBox.warning(
                self, "PDF Export", "Please select at least 2 sensors for comparison."
            )
            return

        output_path = self.file_path_input.text().strip()
        if not output_path:
            QMessageBox.warning(
                self, "PDF Export", "Please select an output file path."
            )
            return

        # Check if file exists and confirm overwrite
        if os.path.exists(output_path):
            reply = QMessageBox.question(
                self,
                "File Exists",
                f"The file '{os.path.basename(output_path)}' already exists.\n\n"
                "Do you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # Prepare export configuration
        quality_map = {
            "Standard (300 DPI)": 300,
            "High (600 DPI)": 600,
            "Print Ready (1200 DPI)": 1200,
        }

        config = {
            "title": self.title_input.text() or "Sensor Comparison Report",
            "include_bar_charts": self.include_bar_charts.isChecked(),
            "include_radar_charts": self.include_radar_charts.isChecked(),
            "include_analysis": self.include_analysis.isChecked(),
            "include_recommendations": self.include_recommendations.isChecked(),
            "notes": self.notes_input.toPlainText(),
            "export_quality": quality_map.get(self.quality_combo.currentText(), 300),
            "page_format": self.format_combo.currentText(),
            "export_type": "Modern Report",
        }

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.generate_button.setEnabled(False)
        self.status_label.setText("Generating PDF report...")

        # Start PDF generation in background
        self.pdf_thread = PDFGenerationThread(self.sensors_data, config, output_path)
        self.pdf_thread.finished.connect(self.on_pdf_generated)
        self.pdf_thread.error.connect(self.on_pdf_error)
        self.pdf_thread.progress.connect(self.progress_bar.setValue)
        self.pdf_thread.start()

    def on_pdf_generated(self, pdf_path):
        """Handle successful PDF generation."""
        self.progress_bar.setVisible(False)
        self.generate_button.setEnabled(True)
        self.open_button.setEnabled(True)

        self.generated_pdf_path = pdf_path
        self.status_label.setText(
            f"PDF report generated successfully: {os.path.basename(pdf_path)}"
        )

        # Create custom message box with Open PDF option
        msg = QMessageBox(self)
        msg.setWindowTitle("PDF Export Complete")
        msg.setText("PDF report has been generated successfully!")
        msg.setDetailedText(f"Saved to: {pdf_path}")
        msg.setIcon(QMessageBox.Information)

        # Add custom buttons
        open_button = msg.addButton("Open PDF", QMessageBox.AcceptRole)
        msg.addButton("Close", QMessageBox.RejectRole)
        msg.setDefaultButton(open_button)

        # Execute dialog and handle result
        msg.exec()
        if msg.clickedButton() == open_button:
            self.open_generated_report()

        logger.info(f"PDF report generated: {pdf_path}")

    def on_pdf_error(self, error_message):
        """Handle PDF generation error."""
        self.progress_bar.setVisible(False)
        self.generate_button.setEnabled(True)

        self.status_label.setText(
            "PDF generation failed. Please check the configuration and try again."
        )

        QMessageBox.critical(
            self,
            "PDF Export Error",
            f"Failed to generate PDF report:\n\n{error_message}",
        )

        logger.error(f"PDF generation error: {error_message}")

    def open_generated_report(self):
        """Open the generated PDF report."""
        if hasattr(self, "generated_pdf_path") and os.path.exists(
            self.generated_pdf_path
        ):
            try:
                import platform
                import subprocess

                if platform.system() == "Linux":
                    subprocess.run(["xdg-open", self.generated_pdf_path])
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", self.generated_pdf_path])
                elif platform.system() == "Windows":
                    os.startfile(self.generated_pdf_path)
                else:
                    QMessageBox.information(
                        self, "Open Report", f"PDF saved to: {self.generated_pdf_path}"
                    )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Open Report",
                    f"Could not open PDF automatically.\nFile location: {self.generated_pdf_path}\n\nError: {e}",
                )
        else:
            QMessageBox.warning(
                self, "Open Report", "No PDF report has been generated yet."
            )

    def select_all_charts(self):
        """Select all chart types for export."""
        self.include_bar_charts.setChecked(True)
        self.include_radar_charts.setChecked(True)

    def select_no_charts(self):
        """Deselect all chart types."""
        self.include_bar_charts.setChecked(False)
        self.include_radar_charts.setChecked(False)
