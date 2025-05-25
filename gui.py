import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import os
import sys
import webbrowser
import pandas as pd
from routing import (
    load_scats_sites,
    build_road_graph,
    snap_sites_to_graph,
    load_predicted_volumes,
    compute_travel_time_weights,
    run_all_algorithms,
    print_route_summary,
    save_multi_route_map,
)


class StdoutRedirector:
    def __init__(self, printer): self._printer = printer
    def write(self, msg): self._printer(msg.strip())
    def flush(self): pass


class TrafficFlowGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Traffic Flow GUI")
        self.geometry("1000x750")
        self.resizable(True, True)

        self.bg_img = None
        self.bg_label = None
        self.sites = None
        self.snapped_sites = None
        self.G = None

        # Redirect stdout/stderr to GUI
        sys.stdout = StdoutRedirector(self._print)
        sys.stderr = StdoutRedirector(self._print)

        self._bg()
        self._form()
        self._results()
        self._status_bar()

        self._load_scats_data()

    def _bg(self):
        bg_path = "assets/bg.jpg"
        if not os.path.exists(bg_path):
            return
        img = Image.open(bg_path).resize((1000, 750))
        self.bg_img = ImageTk.PhotoImage(img)
        self.bg_label = tk.Label(self, image=self.bg_img)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    def _form(self):
        frame = ttk.LabelFrame(self, text="Input")
        frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame, text="Start SCATS ID:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.start_entry = ttk.Combobox(frame, values=[], width=18)
        self.start_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="End SCATS ID:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.end_entry = ttk.Combobox(frame, values=[], width=18)
        self.end_entry.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frame, text="Algorithms:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.algo_list = tk.Listbox(frame, selectmode="multiple", height=6, exportselection=False)
        for algo in ["astar", "bfs", "dfs", "gbfs", "dijkstra", "landmark_astar"]:
            self.algo_list.insert("end", algo)
        self.algo_list.grid(row=0, column=5, padx=5, pady=5)

        self.run_button = ttk.Button(frame, text="Run Routing", command=self._on_run)
        self.run_button.grid(row=0, column=6, padx=10, pady=5)

        self.clear_button = ttk.Button(frame, text="Clear Results", command=self._on_clear)
        self.clear_button.grid(row=0, column=7, padx=5, pady=5)

        self.export_button = ttk.Button(frame, text="Export Summary", command=self._on_export)
        self.export_button.grid(row=0, column=8, padx=5, pady=5)

    def _results(self):
        frame = ttk.LabelFrame(self, text="Results")
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.result_box = ScrolledText(frame, wrap="word", height=20)
        self.result_box.pack(fill="both", expand=True)

    def _status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        bar = ttk.Label(self, textvariable=self.status_var, anchor="w")
        bar.pack(side="bottom", fill="x")

    def _print(self, msg):
        if msg.strip():
            self.result_box.insert("end", msg + "\n")
            self.result_box.see("end")

    def _set_status(self, msg):
        self.status_var.set(msg)

    def _set_buttons_state(self, state):
        self.run_button["state"] = state
        self.clear_button["state"] = state
        self.export_button["state"] = state

    def _on_clear(self):
        self.result_box.delete(1.0, "end")
        self._set_status("Results cleared.")

    def _on_export(self):
        if not hasattr(self, "latest_results"):
            self._print("[WARN] No routing results to export.")
            return
        df = pd.DataFrame(self.latest_results)
        export_path = "output/routes/route_summary.csv"
        df.to_csv(export_path, index=False)
        self._print(f"[INFO] Summary exported to: {export_path}")

    def _load_scats_data(self):
        self._print("Loading SCATS site and network data...")
        csv_path = "output/Scats_Data_October_2006_parsed.csv"
        predicted_csv = "output/predicted/gru_site_predictions.csv"

        if not os.path.exists(predicted_csv):
            self._print("[ERROR] Missing predicted volume file.")
            self._set_status("Failed to load")
            return

        self.sites = load_scats_sites(csv_path)
        self.G = build_road_graph(self.sites)
        self.snapped_sites = snap_sites_to_graph(self.G, self.sites)
        volume_map = load_predicted_volumes(predicted_csv)
        volume_by_node = {
            row["nearest_node"]: volume_map.get(row["SCATS"], 0)
            for _, row in self.snapped_sites.iterrows()
        }
        compute_travel_time_weights(self.G, volume_by_node)
        self._print("SCATS data loaded successfully.")

        ids = self.sites['SCATS'].astype(str).drop_duplicates().tolist()
        self.start_entry['values'] = ids
        self.end_entry['values'] = ids

        self._print("\nAvailable SCATS Sites list:")
        for _, row in self.sites[['SCATS', 'Location']].drop_duplicates().iterrows():
            self._print(f"{int(row.SCATS)}: {row.Location}")
        self._set_status("Ready")

    def _on_run(self):
        start_id = self.start_entry.get().strip()
        end_id = self.end_entry.get().strip()

        if not start_id or not end_id:
            self._print("[INFO] No input provided. Running all algorithms using default IDs.")
            self._print("[TIP] To run a custom route, enter start and end SCATS IDs above.")
            start_id = "970"
            end_id = "4821"

        selected_algos = [self.algo_list.get(i) for i in self.algo_list.curselection()]
        algos = selected_algos if selected_algos else None

        self._set_status("Running routing algorithms...")
        self._set_buttons_state("disabled")

        results = run_all_algorithms(self.G, self.snapped_sites, start_id, end_id, algos=algos)

        if not results:
            self._print("[ERROR] No routes found. Check SCATS IDs.")
            self._set_status("Failed")
            self._set_buttons_state("normal")
            return

        self.latest_results = []
        for r in results:
            self._print(f"{r['algo'].upper()}: {r['time_min']:.2f} min")
            out_path = os.path.join("output/routes", f"segment_times_{r['algo']}_{start_id}_to_{end_id}.png")
            print_route_summary(r['route'], self.G, self.snapped_sites, save_path=out_path)
            self.latest_results.append({"Algorithm": r['algo'].upper(), "TravelTimeMin": round(r['time_min'], 2)})

        best = min(results, key=lambda r: r['time_min'])
        self._print(f"\n[RESULT] Best route: {best['algo'].upper()} – {best['time_min']:.2f} min")

        save_multi_route_map(self.G, results, self.snapped_sites, start_id, end_id)
        map_file = os.path.join("output/routes", f"multi_route_map_{start_id}_to_{end_id}.html")
        webbrowser.open(map_file)

        self._print("\n[INFO] Map opened in browser. Charts saved to output folder.")
        self._set_status("Done")
        self._set_buttons_state("normal")


if __name__ == "__main__":
    app = TrafficFlowGUI()
    app.mainloop()
