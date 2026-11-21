"""Chinese desktop UI for the Linux worker, also usable through WSLg."""
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .jobs import OUTPUT, ROOT, read

STAGES = {"reconstruct": "相机配准与稀疏重建", "segment": "逐图语义分割", "export": "融合、聚类与 IFC 校验"}
STATUS = {"ready": "就绪", "running": "处理中", "complete": "已完成", "failed": "失败"}


class Studio:
    def __init__(self, window):
        self.window = window
        window.title("AutoBIM Studio · 照片到 IFC")
        window.geometry("900x620")
        panel = ttk.Frame(window, padding=16)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(6, weight=1)
        self.photos = tk.StringVar()
        default_model = ROOT.parent / "acceptance" / "model2.onnx"
        self.model = tk.StringVar(value=os.getenv("AUTOBIM_MODEL", str(default_model) if default_model.is_file() else ""))
        for row, (title, variable, browse) in enumerate([
            ("场景照片目录", self.photos, filedialog.askdirectory),
            ("SegFormer ONNX", self.model, lambda: filedialog.askopenfilename(filetypes=[("ONNX", "*.onnx")]))
        ]):
            ttk.Label(panel, text=title).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(panel, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8)
            ttk.Button(panel, text="选择", command=lambda v=variable, b=browse: self.choose(v,b)).grid(row=row,column=2)
        ttk.Button(panel,text="从照片创建项目并运行",command=self.start).grid(row=2,column=1,sticky="w",pady=8)
        ttk.Label(panel,text="项目记录").grid(row=3,column=0,sticky="w")
        self.projects=ttk.Combobox(panel,state="readonly")
        self.projects.grid(row=3,column=1,sticky="ew",padx=8)
        ttk.Button(panel,text="继续该项目",command=self.resume).grid(row=3,column=2)
        self.status=tk.StringVar()
        ttk.Label(panel,textvariable=self.status).grid(row=4,column=0,columnspan=3,sticky="w",pady=8)
        ttk.Button(panel,text="打开项目目录",command=self.open_folder).grid(row=5,column=0,sticky="w")
        self.log=tk.Text(panel,wrap="word",state="disabled")
        self.log.grid(row=6,column=0,columnspan=3,sticky="nsew",pady=8)
        self.process=None
        self.previous=None
        self.note=""
        self.refresh()

    @staticmethod
    def choose(variable, browse):
        value=browse()
        if value:
            variable.set(value)

    def launch(self,args):
        if self.process and self.process.poll() is None:
            messagebox.showinfo("正在处理","当前窗口已有运行中的任务")
            return
        OUTPUT.mkdir(parents=True,exist_ok=True)
        self.launch_log=OUTPUT/"launcher-{}.log".format(os.getpid())
        with self.launch_log.open("w") as log:
            self.process=subprocess.Popen([sys.executable,"-u","-m","autobim_studio.jobs",*args],
                                          cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.note="正在检查项目输入"

    def start(self):
        if not self.photos.get().strip() or not self.model.get().strip():
            messagebox.showinfo("项目输入","请选择照片目录和 ONNX 模型")
            return
        self.launch(["run",self.photos.get(),self.model.get()])

    def resume(self):
        if self.projects.get():
            self.launch(["resume",str(OUTPUT/self.projects.get())])

    def open_folder(self):
        job=OUTPUT/self.projects.get()
        if job.is_dir():
            if os.getenv("WSL_DISTRO_NAME"):
                path=subprocess.check_output(["wslpath","-w",str(job)],text=True).strip()
                subprocess.Popen(["explorer.exe",path])
            else:
                subprocess.Popen(["xdg-open",str(job)])

    def refresh(self):
        names=sorted((p.parent.name for p in OUTPUT.glob("*/project.json")),reverse=True)
        if tuple(names)!=tuple(self.projects["values"] or ()):
            self.projects["values"]=names
            if names:
                self.projects.set(names[0])
        if self.process and self.process.poll() is not None:
            if self.process.returncode:
                messagebox.showerror("任务未完成",self.launch_log.read_text(errors="replace")[-1500:])
            self.process=None
            self.note=""
        job=OUTPUT/self.projects.get()
        if self.projects.get():
            state=read(job)
            self.status.set(" · ".join(filter(None,[STATUS[state["status"]],STAGES.get(state["stage"]),state["error"]])))
            logfile=job/"pipeline.log"
            if logfile.exists():
                key=(job,logfile.stat().st_size)
                if key!=self.previous:
                    with logfile.open("rb") as stream:
                        stream.seek(max(0,key[1]-12000))
                        content=stream.read().decode(errors="replace")
                    self.log.configure(state="normal")
                    self.log.delete("1.0","end")
                    self.log.insert("end",content)
                    self.log.see("end")
                    self.log.configure(state="disabled")
                    self.previous=key
        else:
            self.status.set(self.note or "请选择同一场景的 JPEG 照片")
        self.window.after(1000,self.refresh)


if __name__=="__main__":
    root=tk.Tk()
    Studio(root)
    root.mainloop()
