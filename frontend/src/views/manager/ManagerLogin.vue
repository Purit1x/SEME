<template>
    <div class="manager-login">
        <div class="login-container">
            <div class="login-box">
                <div class="login-header">
                    <h2>MRI预测系统管理后台</h2>
                    <p>Management System</p>
                </div>
                <el-form
                    ref="loginFormRef"
                    :model="loginForm"
                    :rules="loginRules"
                    class="login-form"
                >
                    <el-form-item prop="admin_id">
                        <el-input
                            v-model="loginForm.admin_id"
                            placeholder="请输入管理员账号"
                            size="large"
                        >
                            <template #prefix>
                                <el-icon><User /></el-icon>
                            </template>
                        </el-input>
                    </el-form-item>
                    <el-form-item prop="password">
                        <el-input
                            v-model="loginForm.password"
                            type="password"
                            placeholder="请输入密码"
                            show-password
                            size="large"
                            @keyup.enter="handleLogin"
                        >
                            <template #prefix>
                                <el-icon><Lock /></el-icon>
                            </template>
                        </el-input>
                    </el-form-item>
                    <el-form-item>
                        <el-button
                            type="primary"
                            :loading="loading"
                            class="login-button"
                            size="large"
                            @click="handleLogin"
                        >
                            登录
                        </el-button>
                    </el-form-item>
                </el-form>
                <div class="login-footer">
                    <p>Copyright © 2024 MRI预测系统</p>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'
import axios from 'axios'

const router = useRouter()
const loading = ref(false)
const loginFormRef = ref(null)

const loginForm = ref({
    admin_id: '',
    password: ''
})

const loginRules = {
    admin_id: [
        { required: true, message: '请输入管理员账号', trigger: 'blur' },
        { type: 'string', pattern: /^\d+$/, message: '账号必须为数字', trigger: 'blur' }
    ],
    password: [
        { required: true, message: '请输入密码', trigger: 'blur' },
        { min: 6, message: '密码长度不能小于6位', trigger: 'blur' }
    ]
}

const handleLogin = async () => {
    if (!loginFormRef.value) return

    try {
        await loginFormRef.value.validate()
        loading.value = true

        const requestData = {
            admin_id: parseInt(loginForm.value.admin_id),
            password: loginForm.value.password
        }
        
        console.log('管理员登录请求数据:', requestData)

        const response = await axios.post(
            'http://localhost:5000/api/auth/admin/login', 
            requestData,
            {
                headers: {
                    'Content-Type': 'application/json'
                }
            }
        )

        // 输出完整的响应数据结构
        console.log('响应状态:', response.status)
        console.log('响应数据:', {
            完整响应: response.data,
            success: response.data.success,
            message: response.data.message,
            data: response.data.data,
            admin: response.data.admin
        })

        // 检查响应中是否包含token
        if (response.data && response.data.access_token) {
            // 保存管理员信息
            const adminInfo = {
                admin_id: response.data.admin_id,
                username: response.data.username,
                access_token: response.data.access_token,
                userType: 'admin'
            }
            console.log('保存的管理员信息:', adminInfo)
            
            localStorage.setItem('access_token', response.data.access_token)
            localStorage.setItem('user_type', 'admin')
            localStorage.setItem('admin_info', JSON.stringify(adminInfo))
            
            // 验证保存的信息
            console.log('localStorage中的数据:', {
                access_token: localStorage.getItem('access_token'),
                userType: localStorage.getItem('user_type'),
                adminInfo: JSON.parse(localStorage.getItem('admin_info'))
            })
            
            ElMessage.success('登录成功')
            router.push('/manager/home')
        } else {
            console.error('登录响应中缺少token:', response.data)
            ElMessage.error('登录失败：无法获取访问令牌')
        }
    } catch (error) {
        console.error('登录失败，完整错误信息:', error)
        console.error('错误响应数据:', error.response?.data)
        console.error('错误状态码:', error.response?.status)
        
        if (error.response?.status === 401) {
            ElMessage.error('账号或密码错误')
        } else if (error.response?.data?.message) {
            ElMessage.error(error.response.data.message)
        } else {
            ElMessage.error('登录失败，请稍后重试')
        }
    } finally {
        loading.value = false
    }
}
</script>

<style scoped>
.manager-login {
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background: linear-gradient(135deg, #1890ff 0%, #36cfc9 100%);
    position: relative;
    overflow: hidden;
}

.login-container {
    width: 100%;
    max-width: 440px;
    padding: 20px;
    position: relative;
    z-index: 1;
}

.login-box {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 40px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

.login-header {
    text-align: center;
    margin-bottom: 40px;
}

.login-header h2 {
    font-size: 26px;
    background: linear-gradient(90deg, #1890ff, #36cfc9);
    -webkit-background-clip: text;
    color: transparent;
    margin: 0 0 12px 0;
    font-weight: 600;
}

.login-header p {
    color: #909399;
    font-size: 15px;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.8;
}

.login-form {
    margin-bottom: 24px;

    :deep(.el-input) {
        --el-input-height: 46px;
        margin-bottom: 8px;
    }

    :deep(.el-input__wrapper) {
        background: rgba(255, 255, 255, 0.8);
        box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding-left: 16px;
        transition: all 0.3s ease;

        &:hover {
            box-shadow: 0 0 0 1px rgba(24, 144, 255, 0.3);
            background: rgba(255, 255, 255, 0.95);
        }

        &.is-focus {
            background: #fff;
            box-shadow: 0 0 0 1px #1890ff;
        }
    }

    :deep(.el-input__prefix) {
        font-size: 18px;
        color: #909399;
        margin-right: 8px;
    }
}

.login-button {
    width: 100%;
    height: 46px;
    font-size: 16px;
    font-weight: 500;
    letter-spacing: 1px;
    margin-top: 16px;
    background: linear-gradient(90deg, #1890ff 0%, #36cfc9 100%);
    border: none;
    border-radius: 8px;
    transition: all 0.3s ease;

    &:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(24, 144, 255, 0.2);
        opacity: 0.9;
    }

    &:active {
        transform: translateY(0);
    }
}

.login-footer {
    text-align: center;
    margin-top: 32px;
    color: rgba(144, 147, 153, 0.8);
    font-size: 13px;
}

:deep(.el-form-item__error) {
    padding-top: 4px;
    color: rgba(245, 108, 108, 0.9);
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.login-box {
    animation: fadeIn 0.6s ease-out;
}
</style> 