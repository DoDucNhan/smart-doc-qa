'use client';
import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

// Define the User type
interface User {
  id: number;
  username: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  register: (username: string, password: string, email: string) => Promise<boolean>;
  logout: () => void;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const savedToken = localStorage.getItem('token');
    if (savedToken) {
      setToken(savedToken);
      axios.defaults.headers.common['Authorization'] = `Token ${savedToken}`;
    }
    setIsLoading(false);
  }, []);

  const login = async (username: string, password: string): Promise<boolean> => {
    try {
      const response = await axios.post('http://localhost:8000/auth/login/', {
        username,
        password,
      });
      
      const { token: newToken, user_id } = response.data;
      setToken(newToken);
      setUser({ id: user_id, username });
      localStorage.setItem('token', newToken);
      axios.defaults.headers.common['Authorization'] = `Token ${newToken}`;
      
      return true;
    } catch (err) {
      console.error('Login error:', err);
      return false;
    }
  };

  const register = async (username: string, password: string, email: string): Promise<boolean> => {
    try {
      const response = await axios.post('http://localhost:8000/auth/register/', {
        username,
        password,
        email,
      });
      
      const { token: newToken, user_id } = response.data;
      setToken(newToken);
      setUser({ id: user_id, username });
      localStorage.setItem('token', newToken);
      axios.defaults.headers.common['Authorization'] = `Token ${newToken}`;
      
      return true;
    } catch (err) {
      console.error('Registration error:', err);
      return false;
    }
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('token');
    delete axios.defaults.headers.common['Authorization'];
  };

  return (
    <AuthContext.Provider value={{ user, token, login, register, logout, isLoading }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}