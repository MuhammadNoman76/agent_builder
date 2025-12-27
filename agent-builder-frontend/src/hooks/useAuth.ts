// src/hooks/useAuth.ts
import { useContext, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthContext } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

/**
 * Custom hook for authentication functionality
 * Extends the base AuthContext with additional utilities
 */
export const useAuthHook = () => {
  const context = useContext(AuthContext);
  const navigate = useNavigate();

  if (context === undefined) {
    throw new Error('useAuthHook must be used within an AuthProvider');
  }

  const {
    user,
    isLoading,
    isAuthenticated,
    login: contextLogin,
    register: contextRegister,
    logout: contextLogout,
    refreshUser,
  } = context;

  /**
   * Login with error handling and navigation
   */
  const loginWithRedirect = useCallback(
    async (email: string, password: string, redirectTo: string = '/dashboard') => {
      try {
        await contextLogin(email, password);
        toast.success('Welcome back!');
        navigate(redirectTo);
        return { success: true };
      } catch (error: any) {
        const message = error.response?.data?.detail || 'Login failed. Please try again.';
        toast.error(message);
        return { success: false, error: message };
      }
    },
    [contextLogin, navigate]
  );

  /**
   * Register with error handling and navigation
   */
  const registerWithRedirect = useCallback(
    async (
      username: string,
      email: string,
      password: string,
      redirectTo: string = '/dashboard'
    ) => {
      try {
        await contextRegister(username, email, password);
        toast.success('Account created successfully!');
        navigate(redirectTo);
        return { success: true };
      } catch (error: any) {
        const message =
          error.response?.data?.detail || 'Registration failed. Please try again.';
        toast.error(message);
        return { success: false, error: message };
      }
    },
    [contextRegister, navigate]
  );

  /**
   * Logout with navigation
   */
  const logoutWithRedirect = useCallback(
    (redirectTo: string = '/auth') => {
      contextLogout();
      toast.success('Logged out successfully');
      navigate(redirectTo);
    },
    [contextLogout, navigate]
  );

  /**
   * Check if user has a specific role (for future role-based access)
   */
  const hasRole = useCallback(
    (_role: string): boolean => {
      // Implement role checking logic when roles are added
      return isAuthenticated;
    },
    [isAuthenticated]
  );

  /**
   * Get user's display name
   */
  const displayName = useMemo(() => {
    if (!user) return '';
    return user.username || user.email.split('@')[0];
  }, [user]);

  /**
   * Get user's initials for avatar
   */
  const initials = useMemo(() => {
    if (!user) return '';
    const name = user.username || user.email;
    return name
      .split(/[\s@]/)
      .slice(0, 2)
      .map((part) => part[0]?.toUpperCase() || '')
      .join('');
  }, [user]);

  return {
    // State
    user,
    isLoading,
    isAuthenticated,
    displayName,
    initials,

    // Original actions
    login: contextLogin,
    register: contextRegister,
    logout: contextLogout,
    refreshUser,

    // Extended actions
    loginWithRedirect,
    registerWithRedirect,
    logoutWithRedirect,
    hasRole,
  };
};

export default useAuthHook;
