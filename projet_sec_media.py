"""
=============================================================================
  Mini-Projet : Tatouage Numérique QIM + DCT
=============================================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import cv2
from scipy.fft import dct, idct
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from PIL import Image, ImageTk
import warnings
warnings.filterwarnings("ignore")

# Palette de couleurs
C_BG        = "#0F1117"   # fond principal (noir bleuté)
C_PANEL     = "#1A1D2E"   # fond panneaux
C_CARD      = "#242840"   # fond cartes
C_BORDER    = "#2E3250"   # bordures
C_ACCENT    = "#6C63FF"   # violet accent
C_ACCENT2   = "#00D4AA"   # vert menthe
C_WARN      = "#FF6B6B"   # rouge erreur
C_TEXT      = "#E8E9F3"   # texte principal
C_MUTED     = "#7B7F9E"   # texte secondaire
C_WHITE     = "#FFFFFF"

# Paramètres QIM 
DELTA        = 30
CLE_SECRETE  = 42
TAILLE_BLOC  = 8


# MOTEUR QIM (logique métier)


def dct2d_blocs(image, taille_bloc=8):
    h, w = image.shape
    out = np.zeros_like(image)
    for i in range(0, h - taille_bloc + 1, taille_bloc):
        for j in range(0, w - taille_bloc + 1, taille_bloc):
            b = image[i:i+taille_bloc, j:j+taille_bloc]
            out[i:i+taille_bloc, j:j+taille_bloc] = dct(dct(b, axis=0, norm='ortho'), axis=1, norm='ortho')
    return out

def idct2d_blocs(dct_img, taille_bloc=8):
    h, w = dct_img.shape
    out = np.zeros_like(dct_img)
    for i in range(0, h - taille_bloc + 1, taille_bloc):
        for j in range(0, w - taille_bloc + 1, taille_bloc):
            b = dct_img[i:i+taille_bloc, j:j+taille_bloc]
            out[i:i+taille_bloc, j:j+taille_bloc] = idct(idct(b, axis=1, norm='ortho'), axis=0, norm='ortho')
    return out

def selectionner_positions(shape, nb_bits, cle):
    h, w = shape
    np.random.seed(cle)
    lignes = np.random.randint(h // 5, int(h * 0.6), nb_bits * 3)
    cols   = np.random.randint(w // 5, int(w * 0.6), nb_bits * 3)
    positions = list(set(zip(lignes.tolist(), cols.tolist())))[:nb_bits]
    return positions

def inserer_qim(dct_img, watermark, positions, delta):
    out = dct_img.copy()
    for bit, (i, j) in zip(watermark, positions):
        c = out[i, j]
        out[i, j] = np.round(c / delta) * delta if bit == 0 else (np.floor(c / delta) + 0.5) * delta
    return out

def extraire_qim(image, positions, delta):
    dct_r = dct2d_blocs(image.astype(np.float64), TAILLE_BLOC)
    bits  = np.zeros(len(positions), dtype=int)
    for idx, (i, j) in enumerate(positions):
        c  = dct_r[i, j]
        c0 = np.round(c / delta) * delta
        c1 = (np.floor(c / delta) + 0.5) * delta
        bits[idx] = 0 if abs(c - c0) <= abs(c - c1) else 1
    return bits

def attaque_bruit(image, sigma=10):
    np.random.seed(1)
    return np.clip(image.astype(np.float64) + np.random.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)

def attaque_jpeg(image, qualite=50):
    _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, qualite])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)



# WIDGETS UTILITAIRES


def flat_button(parent, text, command, bg=C_ACCENT, fg=C_WHITE, width=22):
    """Bouton plat avec hover coloré."""
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
        relief="flat", bd=0, cursor="hand2",
        font=("Segoe UI", 10, "bold"), width=width, pady=8
    )
    def on_enter(e): btn.config(bg=_lighten(bg))
    def on_leave(e): btn.config(bg=bg)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn

def _lighten(hex_color):
    """Éclaircit légèrement une couleur hex."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = min(255, r + 30); g = min(255, g + 30); b = min(255, b + 30)
    return f'#{r:02x}{g:02x}{b:02x}'

def label_pair(parent, label, value, row, fg_val=C_ACCENT2):
    """Affiche label + valeur en deux colonnes."""
    tk.Label(parent, text=label, bg=C_CARD, fg=C_MUTED,
             font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", padx=8, pady=2)
    lbl = tk.Label(parent, text=value, bg=C_CARD, fg=fg_val,
                   font=("Consolas", 10, "bold"))
    lbl.grid(row=row, column=1, sticky="e", padx=8, pady=2)
    return lbl



# APPLICATION PRINCIPALE

class WatermarkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tatouage Numérique — QIM + DCT  |  L2-IRS 25/26")
        self.configure(bg=C_BG)
        self.resizable(True, True)
        self.minsize(1000, 680)

        # État interne 
        self.chemin_image   = None
        self.img_originale  = None   # np.ndarray float64
        self.img_tatouee    = None   # np.ndarray uint8
        self.img_bruitee    = None
        self.img_jpeg_atk   = None
        self.watermark      = None
        self.positions      = None
        self.nb_bits_var    = tk.IntVar(value=64)
        self.delta_var      = tk.IntVar(value=30)
        self.sigma_var      = tk.DoubleVar(value=10.0)
        self.qualite_var    = tk.IntVar(value=50)

        # Construction UI 
        self._build_layout()
        self._log("Bienvenue ! Chargez une image pour commencer.", color=C_ACCENT2)

    # LAYOUT 

    def _build_layout(self):
        # Colonne gauche (sidebar) + colonne droite (contenu)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        """Sidebar sombre avec les 3 étapes + paramètres."""
        sidebar = tk.Frame(self, bg=C_PANEL, width=260)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        # Logo / titre
        tk.Label(sidebar, text="🔐", bg=C_PANEL, font=("Segoe UI", 28)).pack(pady=(24, 4))
        tk.Label(sidebar, text="Tatouage Numérique", bg=C_PANEL, fg=C_TEXT,
                 font=("Segoe UI", 13, "bold")).pack()
        tk.Label(sidebar, text="QIM + DCT  •  L2-IRS 25/26", bg=C_PANEL, fg=C_MUTED,
                 font=("Segoe UI", 8)).pack(pady=(2, 18))

        sep = tk.Frame(sidebar, bg=C_BORDER, height=1)
        sep.pack(fill="x", padx=16, pady=4)

        # Étape 1 
        self._sidebar_section(sidebar, "① Charger l'image")
        flat_button(sidebar, "📂  Parcourir…", self._charger_image,
                    bg=C_ACCENT, width=24).pack(padx=18, pady=(6, 2))
        self.lbl_fichier = tk.Label(sidebar, text="Aucune image chargée",
                                    bg=C_PANEL, fg=C_MUTED,
                                    font=("Segoe UI", 8), wraplength=220)
        self.lbl_fichier.pack(padx=18, pady=(0, 12))

        sep2 = tk.Frame(sidebar, bg=C_BORDER, height=1)
        sep2.pack(fill="x", padx=16, pady=4)

        # Paramètres 
        self._sidebar_section(sidebar, "⚙  Paramètres QIM")
        self._param_row(sidebar, "Nombre de bits :", self.nb_bits_var, 8, 256)
        self._param_row(sidebar, "Delta (Δ) :", self.delta_var, 5, 100)
        self._param_row(sidebar, "Sigma bruit :", self.sigma_var, 1, 50)
        self._param_row(sidebar, "Qualité JPEG :", self.qualite_var, 10, 95)

        sep3 = tk.Frame(sidebar, bg=C_BORDER, height=1)
        sep3.pack(fill="x", padx=16, pady=8)

        # Étape 2 
        self._sidebar_section(sidebar, "② Insérer le watermark")
        flat_button(sidebar, "💉  Insérer (aléatoire)", self._inserer,
                    bg="#5A4FD6", width=24).pack(padx=18, pady=(6, 12))

        sep4 = tk.Frame(sidebar, bg=C_BORDER, height=1)
        sep4.pack(fill="x", padx=16, pady=4)

        # Étape 3 
        self._sidebar_section(sidebar, "③ Attaques + Extraction")
        flat_button(sidebar, "⚡  Lancer les attaques", self._attaquer_et_extraire,
                    bg="#C0392B", width=24).pack(padx=18, pady=(6, 12))

        # Bouton reset en bas
        sidebar.pack_propagate(False)
        flat_button(sidebar, "↺  Réinitialiser", self._reset,
                    bg=C_CARD, fg=C_MUTED, width=24).pack(side="bottom", padx=18, pady=16)

    def _sidebar_section(self, parent, title):
        tk.Label(parent, text=title, bg=C_PANEL, fg=C_TEXT,
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x", padx=18, pady=(8, 2))

    def _param_row(self, parent, label, var, from_, to):
        f = tk.Frame(parent, bg=C_PANEL)
        f.pack(fill="x", padx=18, pady=2)
        tk.Label(f, text=label, bg=C_PANEL, fg=C_MUTED,
                 font=("Segoe UI", 8), width=16, anchor="w").pack(side="left")
        entry = tk.Entry(f, textvariable=var, bg=C_CARD, fg=C_TEXT,
                         insertbackground=C_TEXT, relief="flat", width=6,
                         font=("Consolas", 9))
        entry.pack(side="right")

    # MAIN PANEL 

    def _build_main(self):
        main = tk.Frame(self, bg=C_BG)
        main.grid(row=0, column=1, sticky="nsew", padx=0)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        main.rowconfigure(2, weight=0)

        # Barre titre
        header = tk.Frame(main, bg=C_PANEL, height=52)
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(header, text="Visualisation & Résultats",
                 bg=C_PANEL, fg=C_TEXT, font=("Segoe UI", 13, "bold"),
                 anchor="w").pack(side="left", padx=20, pady=12)
        self.lbl_status = tk.Label(header, text="● En attente",
                                   bg=C_PANEL, fg=C_MUTED,
                                   font=("Segoe UI", 9))
        self.lbl_status.pack(side="right", padx=20)

        # Notebook (onglets)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook", background=C_BG, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background=C_CARD, foreground=C_MUTED,
                        padding=[14, 6], font=("Segoe UI", 9))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", C_ACCENT)],
                  foreground=[("selected", C_WHITE)])

        nb = ttk.Notebook(main, style="Dark.TNotebook")
        nb.grid(row=1, column=0, sticky="nsew", padx=12, pady=(10, 4))

        self.tab_images   = tk.Frame(nb, bg=C_BG)
        self.tab_metrics  = tk.Frame(nb, bg=C_BG)
        self.tab_watermark= tk.Frame(nb, bg=C_BG)
        nb.add(self.tab_images,    text="  🖼  Images  ")
        nb.add(self.tab_metrics,   text="  📊  Métriques  ")
        nb.add(self.tab_watermark, text="  🔑  Watermark  ")

        self._build_tab_images()
        self._build_tab_metrics()
        self._build_tab_watermark()

        # Console logs 
        log_frame = tk.Frame(main, bg=C_PANEL, height=130)
        log_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))
        log_frame.grid_propagate(False)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)

        tk.Label(log_frame, text="Console", bg=C_PANEL, fg=C_MUTED,
                 font=("Segoe UI", 8, "bold"), anchor="w").grid(row=0, column=0, sticky="w", padx=10, pady=(6, 0))

        self.console = tk.Text(log_frame, bg=C_PANEL, fg=C_TEXT,
                               font=("Consolas", 8), relief="flat",
                               state="disabled", wrap="word",
                               insertbackground=C_TEXT, height=6)
        self.console.grid(row=1, column=0, sticky="nsew", padx=8, pady=(2, 8))
        sb = tk.Scrollbar(log_frame, command=self.console.yview, bg=C_PANEL)
        sb.grid(row=1, column=1, sticky="ns", pady=(2, 8))
        self.console.config(yscrollcommand=sb.set)
        self.console.tag_config("ok",   foreground=C_ACCENT2)
        self.console.tag_config("err",  foreground=C_WARN)
        self.console.tag_config("info", foreground=C_ACCENT)
        self.console.tag_config("dim",  foreground=C_MUTED)

    # Onglet Images 
    def _build_tab_images(self):
        tab = self.tab_images
        tab.columnconfigure((0, 1, 2, 3), weight=1)
        tab.rowconfigure(1, weight=1)

        labels = ["Image Originale", "Image Tatouée", "Après Bruit", "Après JPEG"]
        colors = [C_MUTED, C_ACCENT, "#F39C12", C_WARN]
        self._img_frames = []
        self._img_labels = []
        self._img_captions = []

        for col, (lbl, col_c) in enumerate(zip(labels, colors)):
            card = tk.Frame(tab, bg=C_CARD, relief="flat")
            card.grid(row=0, column=col, padx=6, pady=10, sticky="nsew")
            card.rowconfigure(1, weight=1)
            card.columnconfigure(0, weight=1)

            # Bandeau couleur en haut
            band = tk.Frame(card, bg=col_c, height=4)
            band.grid(row=0, column=0, sticky="ew")

            # Zone image
            img_lbl = tk.Label(card, bg=C_CARD, text="—\nAucune image",
                               fg=C_MUTED, font=("Segoe UI", 9))
            img_lbl.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")
            self._img_labels.append(img_lbl)

            # Caption
            caption = tk.Label(card, text=lbl, bg=C_CARD, fg=col_c,
                                font=("Segoe UI", 9, "bold"))
            caption.grid(row=2, column=0, pady=(0, 6))

            psnr_lbl = tk.Label(card, text="PSNR : —", bg=C_CARD, fg=C_MUTED,
                                font=("Consolas", 8))
            psnr_lbl.grid(row=3, column=0, pady=(0, 8))
            self._img_captions.append(psnr_lbl)
            self._img_frames.append(card)

    # Onglet Métriques 
    def _build_tab_metrics(self):
        tab = self.tab_metrics
        tab.columnconfigure((0, 1), weight=1)

        # Carte PSNR
        psnr_card = tk.LabelFrame(tab, text="  PSNR — Qualité visuelle (dB)  ",
                                  bg=C_CARD, fg=C_ACCENT,
                                  font=("Segoe UI", 10, "bold"),
                                  relief="flat", bd=1, labelanchor="n")
        psnr_card.grid(row=0, column=0, padx=12, pady=12, sticky="ew")

        self.m_psnr_tattoo = label_pair(psnr_card, "Tatouée vs Originale :", "—", 0)
        self.m_psnr_bruit  = label_pair(psnr_card, "Après bruit vs Originale :", "—", 1)
        self.m_psnr_jpeg   = label_pair(psnr_card, "Après JPEG vs Originale :", "—", 2)

        # Carte BER
        ber_card = tk.LabelFrame(tab, text="  BER — Taux d'erreur binaire  ",
                                 bg=C_CARD, fg=C_ACCENT2,
                                 font=("Segoe UI", 10, "bold"),
                                 relief="flat", bd=1, labelanchor="n")
        ber_card.grid(row=0, column=1, padx=12, pady=12, sticky="ew")

        self.m_ber_direct = label_pair(ber_card, "Sans attaque :", "—", 0)
        self.m_ber_bruit  = label_pair(ber_card, "Après bruit gaussien :", "—", 1)
        self.m_ber_jpeg   = label_pair(ber_card, "Après JPEG :", "—", 2)

        # Légende interprétation
        info = tk.Frame(tab, bg=C_CARD)
        info.grid(row=1, column=0, columnspan=2, padx=12, pady=4, sticky="ew")
        infos = [
            ("PSNR ≥ 40 dB", "Tatouage invisible ✓", C_ACCENT2),
            ("PSNR 30–40 dB", "Qualité acceptable", "#F39C12"),
            ("PSNR < 30 dB", "Dégradation visible ✗", C_WARN),
            ("BER = 0.000", "Extraction parfaite ✓", C_ACCENT2),
            ("BER < 0.100", "Bonne robustesse", "#F39C12"),
            ("BER ≥ 0.500", "Extraction aléatoire ✗", C_WARN),
        ]
        for col, (seuil, desc, color) in enumerate(infos):
            f = tk.Frame(info, bg=C_CARD)
            f.grid(row=0, column=col, padx=8, pady=8)
            tk.Label(f, text=seuil, bg=C_CARD, fg=color,
                     font=("Consolas", 9, "bold")).pack()
            tk.Label(f, text=desc, bg=C_CARD, fg=C_MUTED,
                     font=("Segoe UI", 8)).pack()

    # Onglet Watermark 
    def _build_tab_watermark(self):
        tab = self.tab_watermark
        tab.columnconfigure((0, 1, 2, 3), weight=1)

        titles = ["Watermark inséré", "Extrait (sans attaque)",
                  "Extrait (bruit)", "Extrait (JPEG)"]
        self._wm_text = []

        for col, title in enumerate(titles):
            card = tk.Frame(tab, bg=C_CARD)
            card.grid(row=0, column=col, padx=8, pady=14, sticky="nsew")

            tk.Label(card, text=title, bg=C_CARD, fg=C_TEXT,
                     font=("Segoe UI", 9, "bold")).pack(pady=(10, 4))

            txt = tk.Text(card, bg=C_BG, fg=C_ACCENT2,
                          font=("Consolas", 8), width=20, height=10,
                          relief="flat", state="disabled",
                          wrap="word")
            txt.pack(padx=8, pady=(0, 10), fill="both", expand=True)
            self._wm_text.append(txt)

            ber_lbl = tk.Label(card, text="BER : —", bg=C_CARD,
                               fg=C_MUTED, font=("Consolas", 9))
            ber_lbl.pack(pady=(0, 10))
            self._wm_text.append(ber_lbl)  # index pair = Text, impair = BER label

    # LOGIQUE 

    def _charger_image(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Tous", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Erreur", f"Impossible de lire :\n{path}")
            return

        self.chemin_image  = path
        self.img_originale = img.astype(np.float64)

        # Réinitialisation
        self.img_tatouee = self.img_bruitee = self.img_jpeg_atk = None
        self.watermark   = self.positions = None
        self._reset_ui()

        nom = path.split("/")[-1].split("\\")[-1]
        self.lbl_fichier.config(text=f"✓ {nom}", fg=C_ACCENT2)
        self._afficher_image(0, img)
        self._set_status(f"Image chargée : {img.shape[1]}×{img.shape[0]} px", C_ACCENT2)
        self._log(f"Image chargée : {nom}  ({img.shape[1]}×{img.shape[0]} px)", tag="ok")

    def _inserer(self):
        if self.img_originale is None:
            messagebox.showwarning("Attention", "Veuillez d'abord charger une image.")
            return

        nb_bits = self.nb_bits_var.get()
        delta   = self.delta_var.get()

        self._log(f"Insertion watermark QIM — {nb_bits} bits, Δ={delta} …", tag="info")
        self._set_status("⏳ Insertion en cours…", "#F39C12")

        def _run():
            try:
                wm  = np.random.randint(0, 2, nb_bits)
                dct_orig  = dct2d_blocs(self.img_originale, TAILLE_BLOC)
                positions = selectionner_positions(self.img_originale.shape, nb_bits, CLE_SECRETE)
                dct_tat   = inserer_qim(dct_orig, wm, positions, delta)
                img_tat   = np.clip(idct2d_blocs(dct_tat, TAILLE_BLOC), 0, 255).astype(np.uint8)

                self.watermark    = wm
                self.positions    = positions
                self.img_tatouee  = img_tat

                psnr_val = compute_psnr(self.img_originale.astype(np.uint8), img_tat, data_range=255)

                self.after(0, lambda: self._post_insertion(img_tat, psnr_val, wm))
            except Exception as e:
                self.after(0, lambda: self._log(f"Erreur insertion : {e}", tag="err"))

        threading.Thread(target=_run, daemon=True).start()

    def _post_insertion(self, img_tat, psnr_val, wm):
        self._afficher_image(1, img_tat)
        self.m_psnr_tattoo.config(text=f"{psnr_val:.2f} dB",
                                  fg=C_ACCENT2 if psnr_val >= 40 else "#F39C12")
        self._img_captions[1].config(text=f"PSNR : {psnr_val:.2f} dB")
        self._afficher_wm(0, wm, None, wm)
        self._set_status(f"✓ Watermark inséré — PSNR = {psnr_val:.2f} dB", C_ACCENT2)
        self._log(f"Watermark inséré avec succès. PSNR = {psnr_val:.2f} dB", tag="ok")

    def _attaquer_et_extraire(self):
        if self.img_tatouee is None:
            messagebox.showwarning("Attention", "Veuillez d'abord insérer un watermark.")
            return

        sigma   = self.sigma_var.get()
        qualite = self.qualite_var.get()
        delta   = self.delta_var.get()
        self._log(f"Attaques : bruit σ={sigma}, JPEG q={qualite} …", tag="info")
        self._set_status("⏳ Attaques + extraction…", "#F39C12")

        def _run():
            try:
                img_b = attaque_bruit(self.img_tatouee, sigma)
                img_j = attaque_jpeg(self.img_tatouee, qualite)

                wm_direct = extraire_qim(self.img_tatouee, self.positions, delta)
                wm_bruit  = extraire_qim(img_b, self.positions, delta)
                wm_jpeg   = extraire_qim(img_j, self.positions, delta)

                ber_d = np.mean(self.watermark != wm_direct)
                ber_b = np.mean(self.watermark != wm_bruit)
                ber_j = np.mean(self.watermark != wm_jpeg)

                orig = self.img_originale.astype(np.uint8)
                psnr_b = compute_psnr(orig, img_b, data_range=255)
                psnr_j = compute_psnr(orig, img_j, data_range=255)

                self.img_bruitee  = img_b
                self.img_jpeg_atk = img_j

                self.after(0, lambda: self._post_attaques(
                    img_b, img_j, psnr_b, psnr_j,
                    wm_direct, wm_bruit, wm_jpeg,
                    ber_d, ber_b, ber_j))
            except Exception as e:
                self.after(0, lambda: self._log(f"Erreur attaque : {e}", tag="err"))

        threading.Thread(target=_run, daemon=True).start()

    def _post_attaques(self, img_b, img_j, psnr_b, psnr_j,
                       wm_d, wm_b, wm_j, ber_d, ber_b, ber_j):
        self._afficher_image(2, img_b)
        self._afficher_image(3, img_j)
        self._img_captions[2].config(text=f"PSNR : {psnr_b:.2f} dB")
        self._img_captions[3].config(text=f"PSNR : {psnr_j:.2f} dB")

        def _ber_color(v): return C_ACCENT2 if v < 0.05 else ("#F39C12" if v < 0.2 else C_WARN)
        def _psnr_color(v): return C_ACCENT2 if v >= 40 else ("#F39C12" if v >= 30 else C_WARN)

        self.m_psnr_bruit.config(text=f"{psnr_b:.2f} dB", fg=_psnr_color(psnr_b))
        self.m_psnr_jpeg.config(text=f"{psnr_j:.2f} dB",  fg=_psnr_color(psnr_j))
        self.m_ber_direct.config(text=f"{ber_d:.3f}", fg=_ber_color(ber_d))
        self.m_ber_bruit.config(text=f"{ber_b:.3f}",  fg=_ber_color(ber_b))
        self.m_ber_jpeg.config(text=f"{ber_j:.3f}",   fg=_ber_color(ber_j))

        self._afficher_wm(1, wm_d, ber_d, self.watermark)
        self._afficher_wm(2, wm_b, ber_b, self.watermark)
        self._afficher_wm(3, wm_j, ber_j, self.watermark)

        self._set_status("✓ Attaques terminées — résultats affichés", C_ACCENT2)
        self._log(f"BER sans attaque={ber_d:.3f} | bruit={ber_b:.3f} | JPEG={ber_j:.3f}", tag="ok")
        self._log(f"PSNR bruit={psnr_b:.2f} dB | JPEG={psnr_j:.2f} dB", tag="dim")

    # HELPERS UI 

    def _afficher_image(self, idx, img_arr):
        """Affiche un np.ndarray uint8 dans le label d'image correspondant."""
        h, w = img_arr.shape
        max_dim = 160
        scale = min(max_dim / w, max_dim / h)
        nw, nh = int(w * scale), int(h * scale)
        pil_img = Image.fromarray(img_arr).resize((nw, nh), Image.LANCZOS)
        photo   = ImageTk.PhotoImage(pil_img)
        lbl = self._img_labels[idx]
        lbl.config(image=photo, text="")
        lbl.image = photo  # garder une référence

    def _afficher_wm(self, col, wm_array, ber, wm_ref):
        """Affiche les bits du watermark dans l'onglet Watermark."""
        txt_idx = col * 2
        ber_idx = col * 2 + 1
        if txt_idx >= len(self._wm_text):
            return
        txt_widget = self._wm_text[txt_idx]
        ber_widget = self._wm_text[ber_idx]

        # Formater les bits en grille 8 colonnes
        bits_str = ""
        for i, bit in enumerate(wm_array):
            bits_str += str(bit)
            if (i + 1) % 8 == 0:
                bits_str += "\n"
            else:
                bits_str += " "

        txt_widget.config(state="normal")
        txt_widget.delete("1.0", "end")
        txt_widget.insert("end", bits_str)
        txt_widget.config(state="disabled")

        if ber is not None:
            color = C_ACCENT2 if ber < 0.05 else ("#F39C12" if ber < 0.2 else C_WARN)
            ber_widget.config(text=f"BER : {ber:.3f}", fg=color)

    def _log(self, msg, tag="dim", color=None):
        self.console.config(state="normal")
        self.console.insert("end", f"  › {msg}\n", tag)
        self.console.see("end")
        self.console.config(state="disabled")

    def _set_status(self, msg, color=C_MUTED):
        self.lbl_status.config(text=f"● {msg}", fg=color)

    def _reset_ui(self):
        for lbl in self._img_labels:
            lbl.config(image="", text="—\nAucune image")
        for cap in self._img_captions:
            cap.config(text="PSNR : —")
        for i, w in enumerate(self._wm_text):
            if i % 2 == 0:
                w.config(state="normal"); w.delete("1.0", "end"); w.config(state="disabled")
            else:
                w.config(text="BER : —", fg=C_MUTED)
        for lbl in [self.m_psnr_tattoo, self.m_psnr_bruit, self.m_psnr_jpeg,
                    self.m_ber_direct, self.m_ber_bruit, self.m_ber_jpeg]:
            lbl.config(text="—", fg=C_MUTED)

    def _reset(self):
        self.img_originale = self.img_tatouee = self.img_bruitee = self.img_jpeg_atk = None
        self.watermark = self.positions = self.chemin_image = None
        self.lbl_fichier.config(text="Aucune image chargée", fg=C_MUTED)
        self._reset_ui()
        self._set_status("Réinitialisé", C_MUTED)
        self.console.config(state="normal")
        self.console.delete("1.0", "end")
        self.console.config(state="disabled")
        self._log("Application réinitialisée.", tag="dim")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("version 2")
    app = WatermarkApp()
    app.mainloop()
