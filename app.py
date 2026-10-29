import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Smart Waste Multi-Analytics Ultra", page_icon="♻️", layout="wide"
)

st.title("♻️ Smart Waste Multi-Dimensional Analytics Hub")
st.write(
    "Sistem analisis komprehensif untuk deteksi material, potensi daur ulang, waktu urai, dan instruksi pemilahan sampah."
)


# Load Model YOLOv8
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

# Database Multi-Analisis Lengkap untuk Seluruh Objek COCO / Sampah Harian
EXPANDED_WASTE_DB = {
    # --- PLASTIK, BOTOL & TUMBLER ---
    "bottle": {
        "alias": "Botol / Tumbler / Wadah Minum",
        "kategori": "Plastik & Wadah Daur Ulang",
        "material": "PET / HDPE / Stainless Steel",
        "daur_ulang": "Sangat Tinggi (85% - 95%)",
        "waktu_urai": "450 Tahun (Plastik) / Tidak Urai (Metal)",
        "nilai_ekonomi": "Tinggi (Bank Sampah / Lapak)",
        "bahaya": "Sedang (Mikroplastik jika terbuang sembarangan)",
        "tong": "🔵 Tong Biru (Daur Ulang)",
        "instruksi": "1. Kosongkan sisa cairan.\n2. Jika botol sekali pakai, remas hingga pipih.\n3. Jika tumbler, gunakan kembali.",
    },
    "cup": {
        "alias": "Gelas Plastik / Paper Cup / Cangkir",
        "kategori": "Plastik & Wadah Minuman",
        "material": "Polipropilena (PP) / Kertas Berlapis PE",
        "daur_ulang": "Sedang (50%)",
        "waktu_urai": "20 - 50 Tahun",
        "nilai_ekonomi": "Sedang",
        "bahaya": "Sedang",
        "tong": "🔵 Tong Biru (Daur Ulang)",
        "instruksi": "1. Bilas sisa minuman.\n2. Pisahkan tutup dan sedotan plastik.",
    },
    "bowl": {
        "alias": "Mangkuk / Wadah Makanan / Styrofoam",
        "kategori": "Wadah Plastik / Keramik / Residu",
        "material": "Plastik Rigid / Polistirena / Keramik",
        "daur_ulang": "Rendah (20%)",
        "waktu_urai": "50 - 500 Tahun",
        "nilai_ekonomi": "Rendah",
        "bahaya": "Tinggi (Jika Bahan Styrofoam)",
        "tong": "⚪ Tong Abu-Abu (Residu)",
        "langkah_penanganan": "1. Bersihkan sisa minyak/makanan.\n2. Buang ke kategori sampah residu.",
    },

    # --- KERTAS, BUKU & TISU (MAPPED) ---
    "book": {
        "alias": "Buku / Kertas / Tisu / Kardus Kemasan",
        "kategori": "Kertas & Selulosa",
        "material": "Serat Selulosa Kertas / Pulp",
        "daur_ulang": "Sangat Tinggi (90%)",
        "waktu_urai": "2 - 5 Bulan",
        "nilai_ekonomi": "Sedang (Pengepul Kertas)",
        "bahaya": "Sangat Rendah",
        "tong": "🔵 Tong Biru (Daur Ulang) / 🟢 Organik (Jika Tisu Basah/Kotor)",
        "instruksi": "1. Pastikan dalam kondisi kering jika berupa kertas/buku.\n2. Jika berupa tisu kotor, masukkan ke sampah residu/organik.",
    },

    # --- ELEKTRONIK, MOUSE & GADGET (E-WASTE) ---
    "mouse": {
        "alias": "Mouse Komputer / Peripheral",
        "kategori": "Sampah Elektronik (E-Waste)",
        "material": "Plastik ABS, Kabel Tembaga, PCB Board",
        "daur_ulang": "Khusus (E-Waste Center)",
        "waktu_urai": "Tidak Dapat Urai (>500 Tahun)",
        "nilai_ekonomi": "Tinggi (Komponen Daur Ulang)",
        "bahaya": "Tinggi (Logam Berat & Bahan Kimia)",
        "tong": "🔴 Tong Merah (B3 / E-Waste)",
        "instruksi": "1. Jangan dibakar atau dibuang ke tanah.\n2. Kumpulkan bersama limbah elektronik lainnya ke drop-off point B3.",
    },
    "cell phone": {
        "alias": "Handphone / Kemasan Saset Metalized / Gadget",
        "kategori": "E-Waste / Kemasan Saset Foil",
        "material": "Litium, Logam, Plastik Multilayer Foil",
        "daur_ulang": "Khusus (E-Waste / Ecobrick)",
        "waktu_urai": "100 - 1000 Tahun",
        "nilai_ekonomi": "Tinggi (Jika Gadget) / Rendah (Jika Saset)",
        "bahaya": "Tinggi",
        "tong": "🔴 Tong Merah (E-Waste) / ⚪ Tong Abu-Abu (Residu Saset)",
        "instruksi": "1. Jika hp rusak, pisahkan baterainya.\n2. Jika kemasan saset jajan, bersihkan dan buat ecobrick.",
    },
    "laptop": {
        "alias": "Laptop / Komputer Portabel",
        "kategori": "Sampah Elektronik (E-Waste)",
        "material": "Aluminium, Baterai Li-ion, PCB, Kaca Display",
        "daur_ulang": "Khusus (Pusat E-Waste)",
        "waktu_urai": "Tidak Dapat Urai",
        "nilai_ekonomi": "Sangat Tinggi",
        "bahaya": "Sangat Tinggi (Baterai & Logam Berat)",
        "tong": "🔴 Tong Merah (B3 / E-Waste)",
        "instruksi": "1. Ambil/backup data internal.\n2. Serahkan ke e-waste recycling vendor.",
    },
    "keyboard": {
        "alias": "Keyboard / Papan Ketik Elektronik",
        "kategori": "Sampah Elektronik (E-Waste)",
        "material": "Plastik ABS/PBT, Tembaga, Membran",
        "daur_ulang": "Sedang (Pusat E-Waste)",
        "waktu_urai": ">500 Tahun",
        "nilai_ekonomi": "Sedang",
        "bahaya": "Sedang",
        "tong": "🔴 Tong Merah (B3 / E-Waste)",
        "instruksi": "1. Pisahkan kabel data jika ada.\n2. Masukkan ke kategori daur ulang barang elektronik.",
    },
    "remote": {
        "alias": "Remote Control / Modul Elektronik Kecil",
        "kategori": "Sampah Elektronik (E-Waste)",
        "material": "Plastik Rigid, Baterai AAA/AA, PCB",
        "daur_ulang": "Khusus (E-Waste)",
        "waktu_urai": ">500 Tahun",
        "nilai_ekonomi": "Rendah",
        "bahaya": "Tinggi (Dari Baterai)",
        "tong": "🔴 Tong Merah (B3 / E-Waste)",
        "instruksi": "1. Lepaskan baterai di dalamnya terlebih dahulu.\n2. Buang baterai secara terpisah dari sasis remote.",
    },

    # --- JAJAN, MAKANAN & ORGANIK ---
    "sandwich": {
        "alias": "Jajan / Makanan Kemasan / Roti",
        "kategori": "Sampah Makanan / Organik",
        "material": "Bahan Organik Makanan / Plastik Pembungkus",
        "daur_ulang": "Tinggi (Dapat Dikomposkan)",
        "waktu_urai": "1 - 2 Minggu",
        "nilai_ekonomi": "Rendah (Bahan Kompos/Maggot)",
        "bahaya": "Sangat Rendah",
        "tong": "🟢 Tong Hijau (Organik)",
        "instruksi": "1. Pisahkan makanan dari bungkus plastiknya.\n2. Makanan masuk tong organik, bungkus plastik masuk tong anorganik.",
    },
    "apple": {
        "alias": "Sisa Buah Apel / Organik",
        "kategori": "Sampah Organik",
        "material": "Bahan Organik Buah",
        "daur_ulang": "Sangat Tinggi (Kompos)",
        "waktu_urai": "2 - 4 Minggu",
        "nilai_ekonomi": "Rendah (Pupuk)",
        "bahaya": "Sangat Rendah",
        "tong": "🟢 Tong Hijau (Organik)",
        "instruksi": "1. Buang langsung ke komposter atau maggot box.",
    },
    "banana": {
        "alias": "Kulit Pisang / Organik",
        "kategori": "Sampah Organik",
        "material": "Serat Organik Buah",
        "daur_ulang": "Sangat Tinggi",
        "waktu_urai": "3 - 4 Minggu",
        "nilai_ekonomi": "Rendah",
        "bahaya": "Sangat Rendah",
        "tong": "🟢 Tong Hijau (Organik)",
        "instruksi": "1. Potong kecil-kecil untuk mempercepat pembusukan komposting.",
    },

    # --- PERALATAN LOGAM, KACA & LAINNYA ---
    "scissors": {
        "alias": "Gunting / Alat Potong Logam",
        "kategori": "Logam & Benda Tajam",
        "material": "Stainless Steel / Gagang Plastik",
        "daur_ulang": "Tinggi (80%)",
        "waktu_urai": "100 - 200 Tahun",
        "nilai_ekonomi": "Sedang (Lapak Besi Tua)",
        "bahaya": "Sedang (Risiko Luka Fizik)",
        "tong": "🔵 Tong Biru (Daur Ulang Metal)",
        "instruksi": "1. Bungkus bagian tajam dengan pelindung sebelum dibuang.",
    },
    "toothbrush": {
        "alias": "Sikat Gigi / Plastik Rigid",
        "kategori": "Plastik Residu / Personal Care",
        "material": "Polypropylene & Bulu Nylon",
        "daur_ulang": "Rendah (Sulit Dipisah)",
        "waktu_urai": "500 Tahun",
        "nilai_ekonomi": "Rendah",
        "bahaya": "Sedang",
        "tong": "⚪ Tong Abu-Abu (Residu)",
        "instruksi": "1. Masukkan ke tempat sampah residu umum.",
    },
}

# Fallback Otomatis untuk Objek COCO Lainnya
DEFAULT_WASTE_INFO = {
    "alias": "Objek Anorganik / Kemasan Umum",
    "kategori": "Anorganik / Residu Umum",
    "material": "Komposit Plastik / Sintetis",
    "daur_ulang": "Sedang (30%)",
    "waktu_urai": "50 - 100 Tahun",
    "nilai_ekonomi": "Rendah",
    "bahaya": "Sedang",
    "tong": "⚪ Tong Abu-Abu (Residu)",
    "instruksi": "1. Bersihkan dari sisa kotoran.\n2. Buang ke kategori anorganik umum.",
}

# Sidebar Control
st.sidebar.header("⚙️ Pengaturan Sistem")
input_mode = st.sidebar.radio(
    "Pilih Mode Input:", ("📁 Upload Gambar", "📸 Live Kamera / Webcam")
)

# THRESHOLD DEFAULT DITURUNKAN KE 0.20 AGAR MOUSE & TUMBLER KETANGKAP MUDAH
confidence = st.sidebar.slider("Confidence Threshold (Sensitivitas AI)", 0.05, 1.0, 0.20, 0.05)

uploaded_file = None

if input_mode == "📁 Upload Gambar":
    uploaded_file = st.sidebar.file_uploader(
        "Pilih Gambar...", type=["jpg", "jpeg", "png"]
    )
else:
    st.sidebar.info("Arahkan objek ke kamera lalu ambil foto.")
    uploaded_file = st.camera_input("Ambil Foto dari Kamera")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Sumber")
        st.image(image, use_container_width=True)

    # Proses Deteksi Objek dengan YOLOv8
    with st.spinner("Sedang menganalisis objek & material sampah..."):
        results = model.predict(image, conf=confidence)
        result = results[0]

        # Plot Hasil Bounding Box
        res_plotted = result.plot()
        res_image = Image.fromarray(res_plotted[:, :, ::-1])

    with col2:
        st.subheader("🎯 Hasil Deteksi AI")
        st.image(res_image, use_container_width=True)

    # Analytics Section Multi-Dimensi
    st.markdown("---")
    st.subheader("📊 Analisis Multi-Dimensi Objek Sampah")

    boxes = result.boxes
    if len(boxes) > 0:
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        detected_list = []

        for cls_id, conf in zip(class_ids, confidences):
            raw_name = model.names[cls_id]

            # Abaikan 'person' agar fokus murni ke objek
            if raw_name == "person":
                continue

            # Ambil data analisis komprehensif
            info = EXPANDED_WASTE_DB.get(raw_name, DEFAULT_WASTE_INFO)

            detected_list.append({
                "Nama Objek (COCO)": raw_name.title(),
                "Prediksi Benda": info.get("alias", raw_name.title()),
                "Akurasi AI": f"{conf * 100:.1f}%",
                "Kategori Sampah": info["kategori"],
                "Material Bahan": info["material"],
                "Potensi Daur Ulang": info["daur_ulang"],
                "Estimasi Waktu Urai": info["waktu_urai"],
                "Nilai Ekonomi": info["nilai_ekonomi"],
                "Bahaya Lingkungan": info["bahaya"],
                "Rekomendasi Tong": info["tong"],
                "Instruksi Penanganan": info.get("instruksi", info.get("langkah_penanganan", "-")),
            })

        if len(detected_list) > 0:
            df = pd.DataFrame(detected_list)

            # 1. Metric Executive
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Sampah Terdeteksi", f"{len(df)} Unit")
            m2.metric("Kategori Utama", df["Kategori Sampah"].mode()[0])
            m3.metric("Potensi Daur Ulang", df["Potensi Daur Ulang"].mode()[0])
            m4.metric("Tipe Material Dominan", df["Material Bahan"].mode()[0])

            st.markdown("---")

            # 2. TAB MULTI-ANALISIS INTERAKTIF
            tab1, tab2, tab3 = st.tabs([
                "🔍 Analisis Rinci per Objek",
                "📋 Matriks Tabel Lengkap",
                "📈 Grafik Komposisi",
            ])

            # Tab 1: Expandable Detail Cards
            with tab1:
                st.markdown("### 🧬 Karakteristik & Klasifikasi Objek Terdeteksi")
                for idx, row in df.iterrows():
                    with st.expander(
                        f"📦 Objek #{idx+1}: **{row['Prediksi Benda']}** ({row['Kategori Sampah']}) - Akurasi: {row['Akurasi AI']}",
                        expanded=True,
                    ):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.write(f"**🧪 Material Utama:**\n{row['Material Bahan']}")
                            st.write(f"**⏳ Estimasi Waktu Urai:**\n{row['Estimasi Waktu Urai']}")

                        with c2:
                            st.write(f"**♻️ Potensi Daur Ulang:**\n{row['Potensi Daur Ulang']}")
                            st.write(f"**💵 Nilai Ekonomi:**\n{row['Nilai Ekonomi']}")

                        with c3:
                            st.write(f"**🗑️ Rekomendasi Tempat Sampah:**\n{row['Rekomendasi Tong']}")
                            st.write(f"**⚠️ Risiko Lingkungan:**\n{row['Bahaya Lingkungan']}")

                        st.info(f"**💡 Langkah Penanganan Sesuai Standar:**\n{row['Instruksi Penanganan']}")

            # Tab 2: Matriks Tabel Komprehensif
            with tab2:
                st.markdown("### 📋 Matriks Rincian Komposisi Sampah")
                st.dataframe(
                    df[[
                        "Prediksi Benda",
                        "Kategori Sampah",
                        "Material Bahan",
                        "Potensi Daur Ulang",
                        "Estimasi Waktu Urai",
                        "Rekomendasi Tong",
                        "Akurasi AI",
                    ]],
                    use_container_width=True,
                    hide_index=True,
                )

            # Tab 3: Visualisasi Grafik
            with tab3:
                c_chart1, c_chart2 = st.columns(2)
                with c_chart1:
                    st.markdown("**Grafik Kategori Sampah**")
                    cat_counts = df["Kategori Sampah"].value_counts().reset_index()
                    cat_counts.columns = ["Kategori Sampah", "Jumlah"]
                    st.bar_chart(data=cat_counts, x="Kategori Sampah", y="Jumlah")

                with c_chart2:
                    st.markdown("**Grafik Potensi Daur Ulang**")
                    recycle_counts = df["Potensi Daur Ulang"].value_counts().reset_index()
                    recycle_counts.columns = ["Potensi Daur Ulang", "Jumlah"]
                    st.bar_chart(data=recycle_counts, x="Potensi Daur Ulang", y="Jumlah")

        else:
            st.info("💡 Hanya objek orang ('person') yang terdeteksi. Silakan dekatkan objek sampah, mouse, tumbler, atau bungkus jajan ke kamera.")
    else:
        st.warning("Tidak ada objek yang terdeteksi. Silakan turunkan **Confidence Threshold** di sidebar (misal ke 0.10 - 0.15) atau atur pencahayaan kamera.")
else:
    st.info("👈 Silakan pilih mode input di sidebar (Upload Gambar atau Buka Kamera) untuk mulai!")