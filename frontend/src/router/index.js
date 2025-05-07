import store from '@/stores/index.js'
import {createRouter, createWebHistory} from 'vue-router'

// 使用动态导入
const login = () => import('@/views/doctor/login.vue')
const doctorHome = () => import('@/views/doctor/doctorHome.vue')
const patientManagement = () => import('@/views/doctor/patientManagement.vue')
const effectPrediction = () => import('@/views/doctor/effectPrediction.vue')
const predictionRecords = () => import('@/views/doctor/predictionRecords.vue')
const doctorProfile = () => import('@/views/doctor/doctorProfile.vue')
const patientDetails = () => import('@/views/doctor/patientDetails.vue')
const patientMRI = () => import('@/views/doctor/patientMRI.vue')
const sequenceDetail = () => import('@/views/doctor/sequenceDetail.vue')

const routers = [
  {
    path: '/',
    redirect: (to) => {
      const userType = localStorage.getItem('user_type')
      return userType === 'admin' ? '/manager/home' : '/patient-management'
    }
  },
  {path: '/login',
    name: 'login',
    component: login,
    meta: {title: '登录页面', requiresAuth: false}
  },
  {
    path: '/doctorHome',
    name: 'doctorHome',
    component: doctorHome,
    redirect: '/patient-management',
    children: [
      {
        path: '/patient-management',
        name: 'patientManagement',
        component: patientManagement,
        meta: {title: '患者管理', requiresAuth: true}
      },
      {
        path: '/effect-prediction',
        name: 'effectPrediction',
        component: effectPrediction,
        meta: {title: '效果预测', requiresAuth: true}
      },
      {
        path: '/prediction-records',
        name: 'predictionRecords',
        component: predictionRecords,
        meta: {title: '预测记录', requiresAuth: true}
      },
      {
        path: '/patient-details',
        name: 'patientDetails',
        component: patientDetails,
        meta: {title: '患者详情', requiresAuth: true}
      },
      {
        path: '/patient/:patientId/mri',
        name: 'patientMRI',
        component: patientMRI,
        meta: {title: 'MRI序列管理', requiresAuth: true}
      },
      {
        path: '/patient/:patientId/mri/:sequenceId',
        name: 'sequenceDetail',
        component: sequenceDetail,
        meta: {title: 'MRI序列详情', requiresAuth: true}
      },
      {
        path: '/doctor-profile',
        name: 'doctorProfile',
        component: doctorProfile,
        meta: {title: '个人信息', requiresAuth: true}
      }
    ]
  },
  {
    path: '/manager/login',
    name: 'ManagerLogin',
    component: () => import('../views/manager/ManagerLogin.vue'),
    meta: { requiresAuth: false, userType: 'admin' }
  },
  {
    path: '/manager/home',
    name: 'ManagerHome',
    component: () => import('../views/manager/ManagerHome.vue'),
    meta: { requiresAuth: true, userType: 'admin' },
    children: [
        {
            path: '',
            name: 'ManagerDashboard',
            component: () => import('../views/manager/ManagerDashboard.vue'),
            meta: { requiresAuth: true, userType: 'admin' }
        },
        {
            path: '/manager/patients',
            name: 'ManagerPatients',
            component: () => import('../views/manager/PatientManagement.vue'),
            meta: { requiresAuth: true, userType: 'admin' }
        },
        {
            path: '/manager/doctors',
            name: 'ManagerDoctors',
            component: () => import('../views/manager/DoctorManagement.vue'),
            meta: { requiresAuth: true, userType: 'admin' }
        }
    ]
  }
];

const router = createRouter({history: createWebHistory(), routes: routers});

// 路由守卫
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('access_token')
  const userType = localStorage.getItem('user_type')
  
  console.log('路由守卫:', {
    目标路由: to.path,
    需要认证: to.meta.requiresAuth,
    用户类型: to.meta.userType,
    当前token: token,
    当前用户类型: userType
  })

  if (to.meta.requiresAuth) {
    if (!token) {
      console.log('未登录，重定向到登录页面')
      next({ name: userType === 'admin' ? 'ManagerLogin' : 'Login' })
      return
    }

    if (to.meta.userType && to.meta.userType !== userType) {
      console.log('用户类型不匹配，重定向到登录页面')
      next({ name: userType === 'admin' ? 'ManagerLogin' : 'Login' })
      return
    }
  }

  next()
})

export {router};
