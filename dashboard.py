import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import ttkbootstrap as tb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Global variables
df = None
current_figure = None
original_df = None  # Store original data for filter reset


# --- Export Graph as Image ---
def export_graph():
    """Export the current graph to an image file"""
    global current_figure
    if current_figure is None:
        messagebox.showwarning("Export Warning", "No graph to export. Please generate a graph first.")
        return
    
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", ".png"), ("JPEG Image", ".jpg"), ("PDF Document", "*.pdf")]
        )
        if file_path:
            current_figure.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Export Success", f"Graph saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to export graph:\n{str(e)}")


# ---- Filter/Search Data ----
def filter_table():
    """Filter table data based on search query"""
    global df, original_df
    if df is None:
        return
    
    query = filter_entry.get().strip().lower()
    
    # Clear current table
    for i in tree.get_children():
        tree.delete(i)
    
    if query == '':
        # If query is empty, show original data
        display_df = original_df if original_df is not None else df
        for _, row in display_df.head(100).iterrows():  # Show first 100 rows
            tree.insert("", "end", values=list(row))
        status_label.config(text=f"Showing all data ({len(display_df)} rows)")
    else:
        # Filter data
        try:
            filtered = df[df.apply(lambda row: row.astype(str).str.lower().str.contains(query).any(), axis=1)]
            for _, row in filtered.head(100).iterrows():
                tree.insert("", "end", values=list(row))
            status_label.config(text=f"Filtered: {len(filtered)} rows match")
        except Exception as e:
            messagebox.showerror("Filter Error", f"Failed to filter data:\n{str(e)}")
            status_label.config(text="Filter error")


# ----- Missing Data Visualization ---
def plot_missing_data():
    """Visualize missing data in the dataset"""
    if df is None:
        messagebox.showwarning("No Data", "Please load a dataset first!")
        return
    
    try:
        plt.clf()
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        
        if not missing.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing.plot(kind='bar', color='orange', ax=ax)
            ax.set_title('Missing Values per Column', fontsize=14, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12)
            ax.set_xlabel('Columns', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            status_label.config(text=f"Missing data: {missing.sum()} values")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(["No Missing Data"], [0], color="green")
            ax.set_title("All missing data removed! ✓", fontsize=14, fontweight='bold')
            ax.set_ylabel("Missing Values", fontsize=12)
            plt.tight_layout()
            status_label.config(text="No missing data found")
        
        display_plot(fig)
    except Exception as e:
        messagebox.showerror("Plot Error", f"Failed to plot missing data:\n{str(e)}")


# --- Download Insights ---
def export_insights():
    """Export insights to a text file"""
    try:
        text = insights_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("No Insights", "No insights to export!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text File", ".txt"), ("Markdown File", ".md")]
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            messagebox.showinfo("Export Success", f"Insights saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to export insights:\n{str(e)}")


def drop_duplicates():
    """Remove duplicate and null rows from dataset"""
    global df
    if df is None:
        messagebox.showwarning("No Data", "Please load a dataset first!")
        return
    
    try:
        before = len(df)
        df = df.drop_duplicates()
        df = df.dropna()
        after = len(df)
        
        update_table()
        generate_insights()
        
        # Plot confirmation
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(["No Missing/Duplicate Data"], [0], color="green")
        ax.set_title("All missing and duplicate data removed! ✓", fontsize=14, fontweight='bold')
        ax.set_ylabel("Issues", fontsize=12)
        plt.tight_layout()
        display_plot(fig)
        
        messagebox.showinfo(
            "Rows Removed",
            f"Duplicate and null rows dropped.\nRows before: {before}\nRows after: {after}\nRemoved: {before - after}"
        )
        status_label.config(text=f"Data cleaned: {after} rows remaining")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to drop duplicates:\n{str(e)}")


# --- Update Table (compact preview) ---
def update_table():
    """Update the table preview with current dataframe"""
    for i in tree.get_children():
        tree.delete(i)
    
    if df is not None:
        # Configure columns
        tree["columns"] = list(df.columns)
        tree["show"] = "headings"
        
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center", minwidth=50)
        
        # Display first 100 rows - convert all values to strings to avoid Treeview type conversion issues
        for _, row in df.head(100).iterrows():
            tree.insert("", "end", values=[str(val) if pd.notna(val) else "" for val in row])
        
        status_label.config(text=f"Showing {min(100, len(df))} of {len(df)} rows")


def show_insights_popup():
    """Show insights in a popup window"""
    popup = tk.Toplevel(root)
    popup.title("💡 Smart Insights")
    popup.geometry("600x700")
    popup.transient(root)
    popup.grab_set()
    
    # Scrollbar for text
    scroll = ttk.Scrollbar(popup)
    scroll.pack(side='right', fill='y')
    
    text = tk.Text(popup, height=35, wrap='word', bg="#2a2a2a", fg="white", 
                   relief="flat", yscrollcommand=scroll.set, padx=10, pady=10)
    text.pack(fill='both', expand=True, padx=10, pady=10)
    scroll.config(command=text.yview)
    
    # Fill with current insights
    text.insert(tk.END, insights_text.get(1.0, tk.END))
    text.config(state='disabled')
    
    btn = tb.Button(popup, text="Close", bootstyle="danger", command=popup.destroy)
    btn.pack(pady=10)


# --- Upload Function ---
def upload_file():
    """Upload and load a data file"""
    global df, original_df
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("CSV Files", "*.csv"),
            ("Excel Files", ".xlsx;.xls"),
            ("JSON Files", "*.json"),
            ("All Files", ".")
        ]
    )
    
    if not file_path:
        return
    
    try:
        # Load file based on extension
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            df = pd.read_csv(file_path)
        
        original_df = df.copy()  # Store original for filtering
        
        # Reset selections
        selected_analysis.set("Select")
        selected_column1.set("Select")
        selected_column2.set("Select")
        filter_entry.delete(0, tk.END)
        
        # Update UI
        update_table()
        update_column_selectors()
        generate_insights()
        
        # Show insights popup
        show_insights_popup()
        
        filename = file_path.split('/')[-1]
        status_label.config(text=f"Loaded: {filename} ({len(df)} rows, {len(df.columns)} cols)")
        
    except Exception as e:
        messagebox.showerror("File Load Error", f"Could not load file:\n{str(e)}")
        df = None
        original_df = None


# --- Update Dropdowns ---
def update_column_selectors():
    """Update column selector dropdowns with current dataframe columns"""
    if df is not None:
        columns = ["Select"] + list(df.columns)
        column_selector1["values"] = columns
        column_selector2["values"] = columns
        column_selector1.set("Select")
        column_selector2.set("Select")


# --- Generate Insights ---
def generate_insights():
    """Generate smart insights from the dataset"""
    if df is None:
        return
    
    insights_text.delete(1.0, tk.END)
    insights_text.insert(tk.END, "📊 Smart Insights from Your Data\n")
    insights_text.insert(tk.END, "=" * 50 + "\n\n")
    
    try:
        # General Info
        insights_text.insert(tk.END, f"📋 Dataset Overview:\n")
        insights_text.insert(tk.END, f"   • Records: {len(df):,}\n")
        insights_text.insert(tk.END, f"   • Columns: {len(df.columns)}\n")
        insights_text.insert(tk.END, f"   • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        # Data types
        insights_text.insert(tk.END, f"🔤 Data Types:\n")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            insights_text.insert(tk.END, f"   • {dtype}: {count} columns\n")
        insights_text.insert(tk.END, "\n")
        
        # Missing values
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            insights_text.insert(tk.END, f"⚠  Missing Values: {missing_total:,} total\n")
            missing_cols = df.columns[df.isnull().any()].tolist()
            for col in missing_cols[:5]:  # Show first 5
                count = df[col].isnull().sum()
                pct = (count / len(df)) * 100
                insights_text.insert(tk.END, f"   • {col}: {count:,} ({pct:.1f}%)\n")
            if len(missing_cols) > 5:
                insights_text.insert(tk.END, f"   ... and {len(missing_cols) - 5} more columns\n")
        else:
            insights_text.insert(tk.END, "✓ No missing values found!\n")
        insights_text.insert(tk.END, "\n")
        
        # Duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            pct = (dup_count / len(df)) * 100
            insights_text.insert(tk.END, f"⚠  Duplicate Rows: {dup_count:,} ({pct:.1f}%)\n\n")
        else:
            insights_text.insert(tk.END, "✓ No duplicate rows found!\n\n")
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            insights_text.insert(tk.END, f"📊 Numeric Analysis ({len(numeric_cols)} columns):\n")
            summary = df[numeric_cols].describe().T
            
            # Highest and lowest means
            if len(numeric_cols) > 0:
                top_col = summary['mean'].idxmax()
                low_col = summary['mean'].idxmin()
                insights_text.insert(tk.END, f"   • Highest average: {top_col} ({summary.loc[top_col, 'mean']:.2f})\n")
                insights_text.insert(tk.END, f"   • Lowest average: {low_col} ({summary.loc[low_col, 'mean']:.2f})\n")
            
            # Correlation
            if len(numeric_cols) >= 2:
                try:
                    import numpy as np
                    corr = df[numeric_cols].corr().abs()
                    # Get upper triangle without diagonal
                    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                    upper = corr.where(mask)
                    pairs = upper.unstack().dropna().sort_values(ascending=False)
                    if len(pairs) > 0:
                        top_pair = pairs.index[0]
                        corr_val = pairs.iloc[0]
                        insights_text.insert(tk.END, f"   • Strongest correlation: {top_pair[0]} ↔ {top_pair[1]} ({corr_val:.2f})\n")
                except Exception:
                    pass  # Skip correlation if it fails
            insights_text.insert(tk.END, "\n")
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            insights_text.insert(tk.END, f"📝 Categorical Analysis ({len(cat_cols)} columns):\n")
            for col in cat_cols[:3]:  # Show first 3
                unique = df[col].nunique()
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
                insights_text.insert(tk.END, f"   • {col}: {unique} unique values, most common = '{mode_val}'\n")
            if len(cat_cols) > 3:
                insights_text.insert(tk.END, f"   ... and {len(cat_cols) - 3} more columns\n")
            insights_text.insert(tk.END, "\n")
        
        # Domain-specific insights
        attendance_cols = [col for col in df.columns if "attend" in col.lower()]
        if attendance_cols:
            col = attendance_cols[0]
            if pd.api.types.is_numeric_dtype(df[col]):
                avg_att = df[col].mean()
                insights_text.insert(tk.END, f"📅 Attendance Insight:\n")
                insights_text.insert(tk.END, f"   • Average in '{col}': {avg_att:.2f}%\n\n")
        
        marks_cols = [col for col in df.columns if "mark" in col.lower() or "score" in col.lower()]
        if marks_cols:
            col = marks_cols[0]
            if pd.api.types.is_numeric_dtype(df[col]):
                top_idx = df[col].idxmax()
                top_student = df.loc[top_idx, df.columns[0]]
                top_score = df[col].max()
                insights_text.insert(tk.END, f"🏆 Top Performer:\n")
                insights_text.insert(tk.END, f"   • {top_student} scored {top_score} in '{col}'\n\n")
        
        insights_text.insert(tk.END, "=" * 50 + "\n")
        insights_text.insert(tk.END, "💡 Tip: Use the controls to explore your data further!\n")
        
    except Exception as e:
        insights_text.insert(tk.END, f"\n❌ Error generating insights: {str(e)}\n")


# ----- Generate Graphs ----
def generate_analysis():
    """Generate analysis graph based on selected options"""
    global current_figure
    
    if df is None:
        messagebox.showwarning("No Data", "Please load a dataset first!")
        return
    
    analysis_type = selected_analysis.get()
    col1 = selected_column1.get()
    col2 = selected_column2.get()
    
    if analysis_type == "Select":
        messagebox.showwarning("No Analysis Selected", "Please select an analysis type!")
        return
    
    try:
        plt.close('all')  # Close all previous figures
        
        if analysis_type == "Frequency Analysis":
            if col1 == "Select":
                messagebox.showwarning("Column Required", "Please select a column!")
                return
            fig, ax = plt.subplots(figsize=(10, 6))
            freq = df[col1].value_counts().head(10)
            sns.barplot(x=freq.index.astype(str), y=freq.values, palette="crest", ax=ax)
            ax.set_title(f"Top 10 Frequency of {col1}", fontsize=14, fontweight='bold')
            ax.set_xlabel(col1, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
        elif analysis_type == "Correlation Matrix":
            numeric_df = df.select_dtypes(include='number')
            if numeric_df.empty or len(numeric_df.columns) < 2:
                messagebox.showwarning("No Numeric Data", "Need at least 2 numeric columns for correlation!")
                return
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, 
                       cbar_kws={'label': 'Correlation'})
            ax.set_title("Correlation Matrix", fontsize=14, fontweight='bold')
            
        elif analysis_type == "Marks Summary":
            numeric_df = df.select_dtypes(include='number')
            if numeric_df.empty:
                messagebox.showwarning("No Numeric Data", "No numeric columns found!")
                return
            fig, ax = plt.subplots(figsize=(10, 6))
            summary = numeric_df.describe().T[['mean', 'std', 'min', 'max']]
            summary.plot(kind="bar", ax=ax, legend=True)
            ax.set_title("Statistical Summary of Numeric Columns", fontsize=14, fontweight='bold')
            ax.set_ylabel("Values", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
        elif analysis_type == "Custom Comparison":
            if col1 == "Select" or col2 == "Select":
                messagebox.showwarning("Columns Required", "Please select both columns!")
                return
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=df[col1], y=df[col2], s=60, color="#0dcaf0", 
                           edgecolor="white", alpha=0.7, ax=ax)
            ax.set_title(f"{col1} vs {col2}", fontsize=14, fontweight='bold')
            ax.set_xlabel(col1, fontsize=12)
            ax.set_ylabel(col2, fontsize=12)
            ax.grid(True, alpha=0.3)
            
        elif analysis_type == "Trend Analysis":
            if col1 == "Select":
                messagebox.showwarning("Column Required", "Please select a column!")
                return
            fig, ax = plt.subplots(figsize=(10, 6))
            trend_data = df[col1].value_counts().sort_index()
            trend_data.plot(kind="line", marker='o', ax=ax, linewidth=2, markersize=6)
            ax.set_title(f"Trend of {col1}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Index", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.grid(True, alpha=0.3)
            
        elif analysis_type == "Distribution":
            if col1 == "Select":
                messagebox.showwarning("Column Required", "Please select a column!")
                return
            if not pd.api.types.is_numeric_dtype(df[col1]):
                messagebox.showwarning("Non-numeric Column", "Please select a numeric column for distribution!")
                return
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[col1].dropna(), kde=True, color="#1ad5fa", 
                        edgecolor="white", ax=ax)
            ax.set_title(f"Distribution of {col1}", fontsize=14, fontweight='bold')
            ax.set_xlabel(col1, fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            
        elif analysis_type == "Boxplot - Outliers":
            if col1 == "Select":
                messagebox.showwarning("Column Required", "Please select a column!")
                return
            if not pd.api.types.is_numeric_dtype(df[col1]):
                messagebox.showwarning("Non-numeric Column", "Please select a numeric column for boxplot!")
                return
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=df[col1].dropna(), color="#12e343", ax=ax)
            ax.set_title(f"Boxplot of {col1}", fontsize=14, fontweight='bold')
            ax.set_xlabel(col1, fontsize=12)
            
        elif analysis_type == "Top N Categories":
            if col1 == "Select":
                messagebox.showwarning("Column Required", "Please select a column!")
                return
            n = 10
            n_str = simpledialog.askstring("Top N", "Enter number of top categories (default=10):", 
                                          parent=root, initialvalue="10")
            if n_str and n_str.isdigit() and int(n_str) > 0:
                n = int(n_str)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            top_n = df[col1].value_counts().head(n)
            sns.barplot(y=top_n.index.astype(str), x=top_n.values, palette="crest", ax=ax)
            ax.set_title(f"Top {n} Categories in {col1}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Count", fontsize=12)
            ax.set_ylabel(col1, fontsize=12)
            
        elif analysis_type == "Pairplot":
            numeric_df = df.select_dtypes(include='number')
            if numeric_df.empty or len(numeric_df.columns) < 2:
                messagebox.showwarning("No Numeric Data", "Need at least 2 numeric columns for pairplot!")
                return
            # Limit to first 5 numeric columns for performance
            cols_to_plot = numeric_df.columns[:5].tolist()
            # Sample data if too large (pairplot is slow with large datasets)
            plot_df = df[cols_to_plot]
            if len(plot_df) > 1000:
                plot_df = plot_df.sample(n=1000, random_state=42)
                messagebox.showinfo("Large Dataset", "Sampled 1000 rows for pairplot performance.")
            pairplot = sns.pairplot(plot_df, diag_kind='kde', corner=True)
            pairplot.fig.suptitle("Pairplot of Numeric Columns", y=1.02, fontsize=14, fontweight='bold')
            fig = pairplot.fig
            
        elif analysis_type == "Summary Statistics":
            summary = df.describe(include='all').T
            insights_text.delete(1.0, tk.END)
            insights_text.insert(tk.END, "📊 Summary Statistics\n")
            insights_text.insert(tk.END, "=" * 80 + "\n\n")
            insights_text.insert(tk.END, summary.to_string())
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Summary Statistics\nDisplayed in Insights Panel →", 
                   ha='center', va='center', fontsize=16, fontweight='bold')
            ax.axis('off')
        else:
            messagebox.showwarning("Invalid Selection", "Please select a valid analysis type!")
            return
        
        plt.tight_layout()
        current_figure = fig
        display_plot(fig)
        status_label.config(text=f"Generated: {analysis_type}")
        
    except Exception as e:
        messagebox.showerror("Analysis Error", f"Failed to generate analysis:\n{str(e)}")
        status_label.config(text="Analysis failed")


# ---- Display Plot -----
def display_plot(fig):
    """Display matplotlib figure in the GUI"""
    # Clear previous widgets
    for widget in graph_frame.winfo_children():
        widget.destroy()
    
    # Re-enable propagation temporarily to fit the canvas
    graph_frame.pack_propagate(True)
    
    # Ensure figure is properly sized
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.pack(fill='both', expand=True, padx=5, pady=5)
    
    # Disable propagation again to maintain minimum size
    graph_frame.after(100, lambda: graph_frame.pack_propagate(False))


def on_analysis_change(event=None):
    """Handle analysis type selection change"""
    analysis = selected_analysis.get()
    
    # Reset column selections
    selected_column1.set("Select")
    selected_column2.set("Select")
    
    # Configure column selectors based on analysis type
    if analysis in ["Correlation Matrix", "Marks Summary", "Pairplot", "Summary Statistics"]:
        column_selector1.config(state="disabled")
        column_selector2.config(state="disabled")
    elif analysis in ["Custom Comparison"]:
        column_selector1.config(state="readonly")
        column_selector2.config(state="readonly")
    elif analysis in ["Frequency Analysis", "Trend Analysis", "Distribution", 
                     "Boxplot - Outliers", "Top N Categories"]:
        column_selector1.config(state="readonly")
        column_selector2.config(state="disabled")
    else:
        column_selector1.config(state="readonly")
        column_selector2.config(state="disabled")


# ============================================================================
# GUI SETUP
# ============================================================================

root = tb.Window(themename="darkly")
root.title("EduVis - Educational Data Visualizer")
root.geometry("1400x800")
root.minsize(1000, 600)

# Variables
selected_analysis = tk.StringVar(value="Select")
selected_column1 = tk.StringVar(value="Select")
selected_column2 = tk.StringVar(value="Select")

# --- Title ---
title_label = tb.Label(root, text="📊 EduVis - Educational Data Visualizer", 
                       font=("Helvetica", 18, "bold"))
title_label.pack(pady=10)

upload_btn = tb.Button(root, text="📁 Upload Dataset", bootstyle="success", command=upload_file)
upload_btn.pack(pady=5)

# --- Main Layout ---
main_frame = tb.Frame(root)
main_frame.pack(fill='both', expand=True, padx=10, pady=10)

# === Left Control Panel ===
control_frame = tb.Labelframe(main_frame, text="⚙ Controls", padding=10)
control_frame.pack(side='left', fill='y', padx=(0, 10), pady=5)

ttk.Label(control_frame, text="Analysis Type:").pack(pady=(5, 2))
analysis_menu = ttk.Combobox(control_frame, textvariable=selected_analysis, width=20, 
                            state="readonly", values=[
    "Select", "Frequency Analysis", "Correlation Matrix", "Marks Summary", 
    "Custom Comparison", "Trend Analysis", "Distribution", "Boxplot - Outliers",
    "Top N Categories", "Pairplot", "Summary Statistics"
])
analysis_menu.pack(pady=(0, 10))
analysis_menu.bind("<<ComboboxSelected>>", on_analysis_change)

ttk.Label(control_frame, text="Column 1:").pack(pady=(5, 2))
column_selector1 = ttk.Combobox(control_frame, textvariable=selected_column1, 
                               width=20, state="disabled")
column_selector1.pack(pady=(0, 10))

ttk.Label(control_frame, text="Column 2 (optional):").pack(pady=(5, 2))
column_selector2 = ttk.Combobox(control_frame, textvariable=selected_column2, 
                               width=20, state="disabled")
column_selector2.pack(pady=(0, 15))

generate_btn = tb.Button(control_frame, text="🔍 Generate Graph", bootstyle="info", 
                        command=generate_analysis)
generate_btn.pack(pady=5, fill='x')

export_btn = tb.Button(control_frame, text="💾 Export Graph", bootstyle="warning", 
                      command=export_graph)
export_btn.pack(pady=5, fill='x')

ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

# Filter Section
filter_label = ttk.Label(control_frame, text="🔎 Filter/Search:")
filter_label.pack(pady=(5, 2))

filter_entry = ttk.Entry(control_frame, width=20)
filter_entry.pack(pady=(0, 5), fill='x')

filter_btn = tb.Button(control_frame, text="Apply Filter", bootstyle="primary", 
                      command=filter_table)
filter_btn.pack(pady=5, fill='x')

ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

# Data Actions
missing_data_btn = tb.Button(control_frame, text="📊 Plot Missing Data", 
                            bootstyle="danger", command=plot_missing_data)
missing_data_btn.pack(pady=5, fill='x')

drop_duplicates_btn = tb.Button(control_frame, text="🧹 Clean Data", 
                               bootstyle="danger", command=drop_duplicates)
drop_duplicates_btn.pack(pady=5, fill='x')

export_insights_btn = tb.Button(control_frame, text="📄 Download Insights", 
                               bootstyle="success", command=export_insights)
export_insights_btn.pack(pady=5, fill='x')

ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

status_label = tb.Label(control_frame, text="No file loaded", bootstyle="secondary", 
                       wraplength=180, justify='center')
status_label.pack(pady=10)

# === Center - Graph and Table ===
center_frame = tb.Frame(main_frame)
center_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

# Graph Section (upper part - takes more space)
graph_frame = tb.Labelframe(center_frame, text="📈 Data Visualization", padding=10)
graph_frame.pack(fill='both', expand=True, pady=(0, 10))
graph_frame.pack_propagate(False)  # Prevent frame from shrinking
graph_frame.configure(height=400)  # Set minimum height

placeholder_label = tb.Label(graph_frame, text="📊 Graphs will appear here\n\nLoad a dataset and select an analysis type to begin", 
                             bootstyle="secondary", font=("Helvetica", 12))
placeholder_label.pack(expand=True)

# Table Section (lower part - fixed reasonable size)
table_frame = tb.Labelframe(center_frame, text="📋 Data Preview", padding=10)
table_frame.pack(fill='both', expand=False)
table_frame.pack_propagate(False)  # Prevent frame from shrinking
table_frame.configure(height=250)  # Set fixed height for table

# Scrollbars
tree_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal")
tree_scroll_x.pack(side='bottom', fill='x')

tree_scroll_y = ttk.Scrollbar(table_frame, orient="vertical")
tree_scroll_y.pack(side='right', fill='y')

# Treeview
tree = ttk.Treeview(
    table_frame,
    show='headings',
    xscrollcommand=tree_scroll_x.set,
    yscrollcommand=tree_scroll_y.set,
    height=10
)
tree.pack(fill='both', expand=True)
tree_scroll_x.config(command=tree.xview)
tree_scroll_y.config(command=tree.yview)

# === Right - Insights Panel ===
insights_frame = tb.Labelframe(main_frame, text="💡 Smart Insights", padding=10)
insights_frame.pack(side='right', fill='both', expand=True, pady=5)

insights_scroll = ttk.Scrollbar(insights_frame, orient='vertical')
insights_scroll.pack(side='right', fill='y')

insights_text = tk.Text(insights_frame, height=25, wrap='word', bg="#2a2a2a", 
                       fg="white", relief="flat", padx=10, pady=10,
                       yscrollcommand=insights_scroll.set, font=("Consolas", 10))
insights_text.pack(fill='both', expand=True)
insights_scroll.config(command=insights_text.yview)

# Initial message
insights_text.insert(tk.END, "📊 Welcome to EduVis!\n\n")
insights_text.insert(tk.END, "Upload a dataset to begin exploring your data.\n\n")
insights_text.insert(tk.END, "Supported formats:\n")
insights_text.insert(tk.END, "  • CSV files (.csv)\n")
insights_text.insert(tk.END, "  • Excel files (.xlsx, .xls)\n")
insights_text.insert(tk.END, "  • JSON files (.json)\n")

# Start the application
root.mainloop()

#Faaaaaaaaaaaaaakkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
#iske baad mai tkinter ka project kabhi nhi karunga