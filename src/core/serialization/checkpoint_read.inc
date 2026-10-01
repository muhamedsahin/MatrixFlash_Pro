// Dahili uygulama parçası: src/core/serialization.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Checkpoint okuma, format ve eksik veri denetimi.

Checkpoint Checkpoint::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw IoError("Failed to open file for reading: " + filename);

    uint32_t magic, version;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != MFC_MAGIC || version != 1) {
        throw InvalidArgumentError("Invalid checkpoint format or version: " + filename);
    }

    Checkpoint cp;

    uint32_t num_metadata;
    in.read(reinterpret_cast<char*>(&num_metadata), sizeof(num_metadata));

    for (uint32_t i = 0; i < num_metadata; ++i) {
        uint32_t k_len;
        in.read(reinterpret_cast<char*>(&k_len), sizeof(k_len));
        std::string k(k_len, '\0');
        in.read(&k[0], k_len);

        uint32_t v_len;
        in.read(reinterpret_cast<char*>(&v_len), sizeof(v_len));
        std::string v(v_len, '\0');
        in.read(&v[0], v_len);

        cp.set_metadata(k, v);
    }

    uint32_t num_entries;
    in.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));

    for (uint32_t i = 0; i < num_entries; ++i) {
        uint32_t name_len;
        in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        in.read(&name[0], name_len);

        uint8_t type_val;
        in.read(reinterpret_cast<char*>(&type_val), sizeof(type_val));

        uint32_t num_dims;
        in.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));

        std::vector<uint64_t> dims(num_dims);
        in.read(reinterpret_cast<char*>(dims.data()), num_dims * sizeof(uint64_t));

        Entry entry;
        entry.type = static_cast<EntryType>(type_val);
        entry.shape.resize(num_dims);
        for (uint32_t j = 0; j < num_dims; ++j) {
            entry.shape[j] = static_cast<size_t>(dims[j]);
        }

        uint64_t data_size;
        in.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));

        size_t num_elements = static_cast<size_t>(data_size / sizeof(float));
        entry.data.resize(num_elements);
        in.read(reinterpret_cast<char*>(entry.data.data()), data_size);

        cp.entries_[name] = std::move(entry);
    }

    if (!in.good()) {
        throw IoError("Error reading checkpoint from file: " + filename);
    }

    return cp;
}
