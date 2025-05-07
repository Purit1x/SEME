<template>
    <div class="patient-management">
        <el-card class="patient-list">
            <template #header>
                <div class="card-header">
                    <span>患者列表</span>
                    <el-button type="primary" @click="handleAddPatient">
                        添加患者
                    </el-button>
                </div>
            </template>

            <!-- 搜索栏 -->
            <div class="search-bar">
                <el-input v-model="searchQuery" placeholder="搜索患者姓名/身份证号" clearable @clear="handleSearch"
                    @input="handleSearch">
                    <template #prefix>
                        <el-icon>
                            <Search />
                        </el-icon>
                    </template>
                </el-input>
            </div>

            <!-- 患者列表表格 -->
            <el-table :data="filteredPatientList" style="width: 100%" v-loading="loading">
                <el-table-column prop="id" label="患者ID" width="120" />
                <el-table-column prop="name" label="姓名" width="120" />
                <el-table-column prop="age" label="年龄" width="80" />
                <el-table-column prop="sex" label="性别" width="80" />
                <el-table-column prop="id_number" label="身份证号" width="180" />
                <el-table-column label="操作" fixed="right" width="200">
                    <template #default="scope">
                        <el-button type="primary" link @click="handleView(scope.row)">
                            查看
                        </el-button>
                        <el-button type="primary" link @click="handleEdit(scope.row)">
                            编辑
                        </el-button>

                    </template>
                </el-table-column>
            </el-table>

            <!-- 分页 -->
            <div class="pagination">
                <el-pagination v-model:current-page="currentPage" v-model:page-size="pageSize"
                    :page-sizes="[10, 20, 30, 50]" layout="total, sizes, prev, pager, next, jumper" :total="total"
                    @size-change="handleSizeChange" @current-change="handleCurrentChange" />
            </div>
        </el-card>

        <!-- 添加患者对话框 -->
        <el-dialog v-model="dialogVisible" title="添加患者" width="50%" :before-close="handleClose">
            <el-form ref="patientFormRef" :model="patientForm" :rules="patientRules" label-width="100px">
                <el-form-item label="姓名" prop="patient_name">
                    <el-input v-model="patientForm.patient_name" placeholder="请输入患者姓名" />
                </el-form-item>

                <el-form-item label="性别" prop="sex">
                    <el-radio-group v-model="patientForm.sex">
                        <el-radio label="男">男</el-radio>
                        <el-radio label="女">女</el-radio>
                    </el-radio-group>
                </el-form-item>

                <el-form-item label="年龄" prop="age">
                    <el-input-number v-model="patientForm.age" :min="0" :max="150" placeholder="请输入年龄" />
                </el-form-item>

                <el-form-item label="身份证号" prop="id_number">
                    <el-input v-model="patientForm.id_number" placeholder="请输入身份证号" />
                </el-form-item>

                <el-form-item label="照片" prop="photo">
                    <el-upload class="avatar-uploader" action="#" :auto-upload="false" :show-file-list="false"
                        :on-change="handlePhotoChange">
                        <img v-if="imageUrl" :src="imageUrl" class="avatar" />
                        <el-icon v-else class="avatar-uploader-icon">
                            <Plus />
                        </el-icon>
                    </el-upload>
                </el-form-item>
            </el-form>

            <template #footer>
                <span class="dialog-footer">
                    <el-button @click="dialogVisible = false">取消</el-button>
                    <el-button type="primary" @click="submitPatient">确定</el-button>
                </span>
            </template>
        </el-dialog>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Search, Plus } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'
import { useRouter } from 'vue-router'

// 数据
const router = useRouter()
const loading = ref(false)
const searchQuery = ref('')
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const patientList = ref([])
const filteredPatientList = ref([])

// 添加患者相关
const dialogVisible = ref(false)
const patientFormRef = ref(null)
const imageUrl = ref('')

const patientForm = ref({
    patient_name: '',
    sex: '男',
    age: '',
    id_number: '',
    photo: null
})

const patientRules = {
    patient_name: [
        { required: true, message: '请输入患者姓名', trigger: 'blur' },
        { min: 2, max: 20, message: '长度在 2 到 20 个字符', trigger: 'blur' }
    ],
    sex: [
        { required: true, message: '请选择性别', trigger: 'change' }
    ],
    age: [
        { required: true, message: '请输入年龄', trigger: 'blur' },
        { type: 'number', message: '年龄必须为数字', trigger: 'blur' }
    ],
    id_number: [
        { required: true, message: '请输入身份证号', trigger: 'blur' },
        { pattern: /(^\d{15}$)|(^\d{18}$)|(^\d{17}(\d|X|x)$)/, message: '请输入正确的身份证号', trigger: 'blur' }
    ]
}

// 获取患者列表
const getPatientList = async () => {
    loading.value = true
    try {
        const token = localStorage.getItem('access_token')
        const response = await axios.get('http://localhost:5000/api/patients', {
            headers: {
                'Authorization': `Bearer ${token}`
            },
            params: {
                page: currentPage.value,
                per_page: pageSize.value
            }
        })

        if (response.data.success) {
            patientList.value = response.data.patients
            filteredPatientList.value = patientList.value
            total.value = response.data.pagination.total
            console.log('患者列表数据：', patientList.value)
        }
    } catch (error) {
        ElMessage.error('获取患者列表失败')
        console.error(error)
    } finally {
        loading.value = false
    }
}

// 搜索处理
const handleSearch = () => {
    if (!searchQuery.value) {
        filteredPatientList.value = patientList.value
    } else {
        const query = searchQuery.value.toLowerCase()
        filteredPatientList.value = patientList.value.filter(patient =>
            patient.name.toLowerCase().includes(query) ||
            patient.id_number.toLowerCase().includes(query)
        )
    }
    currentPage.value = 1
    total.value = filteredPatientList.value.length
}

// 分页处理
const handleSizeChange = (val) => {
    pageSize.value = val
    getPatientList()
}

const handleCurrentChange = (val) => {
    currentPage.value = val
    getPatientList()
}

// 处理照片变化
const handlePhotoChange = (file) => {
    const isImage = file.raw.type.startsWith('image/')
    const isLt2M = file.raw.size / 1024 / 1024 < 2

    if (!isImage) {
        ElMessage.error('只能上传图片文件!')
        return
    }
    if (!isLt2M) {
        ElMessage.error('图片大小不能超过 2MB!')
        return
    }

    patientForm.value.photo = file.raw
    imageUrl.value = URL.createObjectURL(file.raw)
}

// 添加患者
const handleAddPatient = () => {
    dialogVisible.value = true
    patientForm.value = {
        patient_name: '',
        sex: '男',
        age: '',
        id_number: '',
        photo: null
    }
    imageUrl.value = ''
}

// 提交患者信息
const submitPatient = async () => {
    if (!patientFormRef.value) return

    await patientFormRef.value.validate(async (valid) => {
        if (valid) {
            try {
                const formData = new FormData()
                formData.append('patient_name', patientForm.value.patient_name)
                formData.append('sex', patientForm.value.sex)
                formData.append('age', patientForm.value.age)
                formData.append('id_number', patientForm.value.id_number)
                if (patientForm.value.photo) {
                    formData.append('photo', patientForm.value.photo)
                }

                const token = localStorage.getItem('access_token')
                const response = await axios.post('http://localhost:5000/api/patients', formData, {
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'multipart/form-data'
                    }
                })

                if (response.data.success) {
                    ElMessage.success('添加患者成功')
                    dialogVisible.value = false
                    getPatientList()
                }
            } catch (error) {
                ElMessage.error('添加患者失败：' + error.response?.data?.message || error.message)
            }
        }
    })
}

// 关闭对话框
const handleClose = (done) => {
    ElMessageBox.confirm('确认关闭？未保存的数据将会丢失')
        .then(() => {
            done()
        })
        .catch(() => { })
}

// 查看患者详情
const handleView = (row) => {
    console.log('要传递的患者原始信息：', {
        姓名: row.name,
        年龄: row.age,
        性别: row.sex,
        身份证号: row.id_number,
        ID: row.id
    })

    const patientData = JSON.stringify(row)
    console.log('JSON字符串：', patientData)
    const encodedData = encodeURIComponent(patientData)
    console.log('编码后的数据：', encodedData)

    router.push({
        name: 'patientDetails',
        query: {
            patientInfo: encodedData
        }
    })
}

// 编辑患者
const handleEdit = (row) => {
    // TODO: 实现编辑患者功能
}

// 删除患者
const handleDelete = (row) => {
    ElMessageBox.confirm(
        '确定要删除该患者吗？',
        '警告',
        {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning',
        }
    ).then(async () => {
        try {
            // TODO: 调用后端API删除患者
            // await axios.delete(`/api/patients/${row.id}`)
            ElMessage.success('删除成功')
            getPatientList()
        } catch (error) {
            ElMessage.error('删除失败')
        }
    }).catch(() => { })
}

onMounted(() => {
    getPatientList()
})
</script>

<style scoped>
.patient-management {
    padding: 20px;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.search-bar {
    margin-bottom: 20px;
}

.pagination {
    margin-top: 20px;
    display: flex;
    justify-content: flex-end;
}

.avatar-uploader {
    border: 1px dashed #d9d9d9;
    border-radius: 6px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    width: 178px;
    height: 178px;
}

.avatar-uploader:hover {
    border-color: #409EFF;
}

.avatar-uploader-icon {
    font-size: 28px;
    color: #8c939d;
    width: 178px;
    height: 178px;
    text-align: center;
    line-height: 178px;
}

.avatar {
    width: 178px;
    height: 178px;
    display: block;
}

.dialog-footer {
    padding: 20px 0 0;
    text-align: right;
}
</style>