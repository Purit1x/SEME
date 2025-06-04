<template>
    <div class="container" id="app">
        <div class="forms-container">
        <div class="signin-signup">
            <el-form
            ref="loginRef"
            :model="loginForm"
            :rules="loginRules"
            class="sign-in-form"
            >
            <h2 class="title">登录</h2>
            <div class="input-field">
                <i class="fa-solid fa-user"></i>
                <el-form-item prop="mail">
                <el-input
                    v-model="loginForm.mail"
                    placeholder="邮箱"
                    @keyup.enter="Login(loginForm)"
                />
                </el-form-item>
            </div>
            <div class="input-field">
                <i class="fa-solid fa-lock"></i>
                <el-form-item prop="password">
                <el-input
                    v-model="loginForm.password"
                    type="password"
                    placeholder="密码"
                    autocomplete="off"
                    show-password
                    @keyup.enter="Login(loginForm)"
                />
                </el-form-item>
            </div>
            <el-button
                type="primary"
                :loading="loginLoading"
                @click="Login(loginForm)"
                class="btn form"
                round
            >
                {{loginLoading ? '登 录 中' : '登 录'}}
            </el-button>
            </el-form>
            <el-form
            ref="signUpRef"
            :model="signUpForm"
            :rules="signUpRules"
            class="sign-up-form"
            >
            <h2 class="title">注册</h2>
            <div class="input-field">
                <i class="fa-solid fa-id-card"></i>
                <el-form-item prop="doctor_id">
                <el-input v-model="signUpForm.doctor_id" placeholder="医生ID" />
                </el-form-item>
            </div>
            <div class="input-field">
                <i class="fa-solid fa-user"></i>
                <el-form-item prop="name">
                <el-input v-model="signUpForm.name" placeholder="姓名" />
                </el-form-item>
            </div>
            <div class="input-field">
                <i class="fa-solid fa-envelope"></i>
                <el-form-item prop="email">
                <el-input v-model="signUpForm.email" placeholder="邮箱" />
                </el-form-item>
            </div>
            <div class="input-field">
                <i class="fa-solid fa-hospital"></i>
                <el-form-item prop="department">
                <el-select v-model="signUpForm.department" placeholder="选择科室" style="width: 100%">
                    <el-option label="放射科" value="放射科" />
                    <el-option label="泌尿外科" value="泌尿外科" />
                    <!-- 可以添加更多科室选项 -->
                </el-select>
                </el-form-item>
            </div>
            <div class="input-field">
                <i class="fa-solid fa-lock"></i>
                <el-form-item prop="password">
                <el-input
                    v-model="signUpForm.password"
                    type="password"
                    placeholder="密码"
                    autocomplete="off"
                    show-password
                />
                </el-form-item>
            </div>
            <div class="input-field">
                <i class="fa-solid fa-lock"></i>
                <el-form-item prop="confirmPassword">
                <el-input
                    v-model="signUpForm.confirmPassword"
                    type="password"
                    placeholder="确认密码"
                    autocomplete="off"
                    show-password
                />
                </el-form-item>
            </div>
            <div class="input-field">
                <i class="fa-solid fa-key"></i>
                <el-form-item prop="verificationCode">
                <el-input v-model="signUpForm.verificationCode" placeholder="验证码">
                    <template #append>
                        <el-button type="primary" @click="SendMail" :loading="sendloading">
                            {{sendloading ? "发送中" : "获取验证码"}}
                        </el-button>
                    </template>
                </el-input>
                </el-form-item>
            </div>
            <el-button
                type="primary"
                :loading="signUploading"
                @click="VerifyCode(signUpForm)"
                class="btn form"
                round
            >
                {{signUploading ? '注 册 中' : '注 册'}}
            </el-button>
            </el-form>
        </div>
        </div>
    
        <div class="panels-container">
        <div class="panel left-panel">
            <div class="content">
            <h3>新用户 ?</h3>
            <p>输入您的信息成为我们的客户</p>
            <button class="btn transparent" id="sign-up-btn">注 册</button>
            </div>
            
        </div>
        <div class="panel right-panel">
            <div class="content">
            <h3>已有账号 ?</h3>
            <p>请登录以享受我们更多的服务</p>
            <button class="btn transparent" id="sign-in-btn">登 录</button>
            </div>
            
        </div>
        </div>
    </div>
    </template>
    
    <style>
    @import '../../assets/doctor/css/style.css';

    .el-button.is-round {
        border-radius: 49px;
    }
    
    .el-form-item {
        margin-bottom: 0;
    }
    
    .el-form-item.is-error .el-input__inner,
    .el-form-item.is-error .el-input__inner:focus,
    .el-form-item.is-error .el-select-v2__wrapper,
    .el-form-item.is-error .el-select-v2__wrapper:focus,
    .el-form-item.is-error .el-textarea__inner,
    .el-form-item.is-error .el-textarea__inner:focus {
        box-shadow: none;
    }
    
    .el-input {
        align-items: center;
        margin-right: 8px;
    }
    
    .el-input .el-input__icon {
        font-size: 1.4em;
    }

    .el-select .el-input {
        width: 100%;
    }

    .input-field .el-select {
        width: 100%;
    }

    .el-input-group__append {
        padding: 0;
    }

    .el-input-group__append button {
        border: none;
        height: 100%;
        border-radius: 0;
    }
    </style>
    
    <script>
    import {ref,onBeforeMount,onMounted} from 'vue';
    import axios  from 'axios';
    import {ElMessage} from 'element-plus';
    import {router} from '../../router';
    
    export default {
        setup() {
            const loginLoading = ref(false);
            const signUploading = ref(false);
            const sendloading = ref(false);
            const isPopupVisible = ref(false);
    
            const loginRef = ref(null);
            const signUpRef = ref(null);
    
            const loginForm = ref({
            mail: "",
            password: "",
            });
    
            const signUpForm = ref({
            doctor_id: "",
            name: "",
            email: "",
            department: "",
            password: "",
            confirmPassword: "",
            verificationCode: ""
            });
    
            const loginRules = {
            mail: [
                {
                required: true,
                message: "请输入邮箱",
                type: "string",
                trigger: "blur",
                },
            ],
            password: [
                {
                required: true,
                message: "请输入密码",
                type: "string",
                trigger: "blur",
                },
            ],
            };
    
            const signUpRules = {
            doctor_id: [
                {
                required: true,
                message: "请输入医生ID",
                trigger: "blur"
                },
                {
                pattern: /^DR\d{3,}$/,
                message: "医生ID格式为DR开头加3位以上数字",
                trigger: "blur"
                }
            ],
            name: [
                {
                required: true,
                message: "请输入姓名",
                trigger: "blur"
                },
                {
                min: 2,
                max: 20,
                message: "姓名长度在2-20个字符之间",
                trigger: "blur"
                }
            ],
            email: [
                {
                required: true,
                message: "请输入邮箱",
                trigger: "blur"
                },
                {
                type: "email",
                message: "请输入正确的邮箱格式",
                trigger: "blur"
                }
            ],
            department: [
                {
                required: true,
                message: "请选择科室",
                trigger: "change"
                }
            ],
            password: [
                {
                required: true,
                message: "请输入密码",
                trigger: "blur"
                },
                {
                pattern: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/,
                message: "密码必须包含大小写字母、数字和特殊字符，且长度不少于8位",
                trigger: "blur"
                }
            ],
            confirmPassword: [
                {
                required: true,
                message: "请确认密码",
                trigger: "blur"
                },
                {
                validator: (rule, value, callback) => {
                    if (value === "") {
                    callback(new Error("请再次输入密码"));
                    } else if (value !== signUpForm.value.password) {
                    callback(new Error("两次输入密码不一致!"));
                    } else {
                    callback();
                    }
                },
                trigger: "blur"
                }
            ],
            verificationCode: [
                {
                required: true,
                message: "请输入验证码",
                trigger: "blur"
                },
                {
                len: 6,
                message: "验证码长度应为6位",
                trigger: "blur"
                }
            ]
            };
    
            onBeforeMount(() => {
            loginLoading.value = false;
            signUploading.value = false;
            const data = localStorage.getItem('microData');
            if (data != null){
                router.push('/patientHome');
            }
            });
    
            onMounted(() => {
            const sign_in_btn = document.querySelector("#sign-in-btn");
            const sign_up_btn = document.querySelector("#sign-up-btn");
            const container = document.querySelector(".container")
    
            sign_in_btn.addEventListener("click", () => {
                container.classList.remove("sign-up-mode");
            });
    
            sign_up_btn.addEventListener("click", () => {
                container.classList.add("sign-up-mode");
            });
            });
    
            const Login = (formData) => {
                const loginApi = "http://localhost:5000/api/auth/doctor/login"
                loginRef.value.validate((valid) => {
                    if (valid) {
                        loginLoading.value = true;
                        const params = {
                            "login_id": loginForm.value.mail,
                            "password": loginForm.value.password
                        };
                        
                        axios.post(loginApi, params, {
                            headers: {
                                'Content-Type': 'application/json'
                            }
                        })
                        .then(response => {
                            if(response.data.success){
                                console.log("登录成功");
                                const { access_token, refresh_token, doctor } = response.data;
                                
                                // 存储登录信息
                                localStorage.setItem('access_token', access_token);
                                localStorage.setItem('refresh_token', refresh_token);
                                localStorage.setItem('doctorInfo', JSON.stringify(doctor));
                                
                                ElMessage.success("登录成功");
                                loginLoading.value = false;
                                
                                // 使用 router.replace 替代 router.push，这样用户不能通过浏览器返回按钮回到登录页
                                router.replace('/doctorHome')
                                    .catch(err => {
                                        console.error('路由跳转失败:', err);
                                        ElMessage.error('页面跳转失败，请重试');
                                    });
                            }
                        })
                        .catch(error => {
                            console.error("登录失败:", error);
                            ElMessage.error(error.response?.data?.message || "登录失败，请检查邮箱和密码");
                            loginForm.value.password = "";
                            loginLoading.value = false;
                        });
                    }
                });
            };
    
            
            const SendMail = async () => {
                if (!signUpForm.value.email || !signUpForm.value.doctor_id || 
                    !signUpForm.value.name || !signUpForm.value.department || 
                    !signUpForm.value.password) {
                    ElMessage.warning("请先填写完整注册信息");
                    return;
                }
    
                try {
                    sendloading.value = true;
                    const registerApi = "http://localhost:5000/api/auth/doctor/register";
                    
                    const params = {
                        "doctor_id": signUpForm.value.doctor_id,
                        "name": signUpForm.value.name,
                        "password": signUpForm.value.password,
                        "email": signUpForm.value.email,
                        "department": signUpForm.value.department
                    };
    
                    const response = await axios.post(registerApi, params, {
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
    
                    if (response.data.success) {
                        localStorage.setItem('verification_id', response.data.verification_id);
                        ElMessage.success("验证码已发送，请查收邮件");
                    }
    
                    setTimeout(() => {
                        sendloading.value = false;
                    }, 1000);
                } catch (error) {
                    sendloading.value = false;
                    ElMessage.error("验证码发送失败：" + error.response?.data?.message || error.message);
                }
            };
    
            const VerifyCode = async (formData) => {
                const verification_id = localStorage.getItem('verification_id');
                
                if (!verification_id) {
                    ElMessage.error("请先获取验证码");
                    return;
                }
    
                try {
                    signUpRef.value.validate(async (valid) => {
                        if (valid) {
                            signUploading.value = true;
                            const verifyApi = "http://localhost:5000/api/auth/doctor/verify";
    
                            const params = {
                                "verification_id": verification_id,
                                "code": signUpForm.value.verificationCode
                            };
    
                            const response = await axios.post(verifyApi, params, {
                                headers: {
                                    'Content-Type': 'application/json'
                                }
                            });
    
                            if (response.data.success) {
                                const { access_token, refresh_token } = response.data;
                                
                                // 存储token
                                localStorage.setItem('access_token', access_token);
                                localStorage.setItem('refresh_token', refresh_token);
                                
                                ElMessage.success("注册成功");
                                signUpRef.value.resetFields();
                                // 清除验证ID
                                localStorage.removeItem('verification_id');
                                // 切换到登录界面
                                document.querySelector("#sign-in-btn").click();
                            }
                            
                            signUploading.value = false;
                        }
                    });
                } catch (error) {
                    signUploading.value = false;
                    ElMessage.error("验证失败：" + error.response?.data?.message || error.message);
                }
            };
    
            const SignUp = (formData) => {
            signUpRef.value.validate((valid) => {
                if (valid) {
                signUploading.value = true;
                // TODO: axios 注册请求
                setTimeout(() => {
                    ElMessage.success("注册成功");
                    signUpRef.value.resetFields();
                    document.getElementById("sign-in-btn").click();
                }, 500);
                }
            });
            signUploading.value = false;
            };
            return {
            loginLoading,
            signUploading,
            sendloading,
            loginRef,
            signUpRef,
            loginForm,
            signUpForm,
            loginRules,
            signUpRules,
            onMounted,
            isPopupVisible,
            Login,
            SignUp,
            SendMail,
            VerifyCode
            };
        }
    }
    </script>