import { NavLink, Outlet, useLocation } from "react-router-dom";
import { FileSearch, Wand2 } from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarInset,
  SidebarProvider,
} from "@/components/ui/sidebar";

const navItems = [
  { label: "Analyze Resume", icon: FileSearch, to: "/analyze" },
  { label: "Improve Resume", icon: Wand2, to: "/improve" },
];

export default function AppLayout() {
  const { pathname } = useLocation();

  return (
    <SidebarProvider className="h-svh">
      <Sidebar collapsible="none">
        <SidebarHeader className="border-b border-sidebar-border">
          <div className="px-2 py-3">
            <h2 className="text-base font-semibold tracking-tight text-sidebar-foreground">
              AIResume
            </h2>
            <p className="text-xs text-sidebar-foreground/60">
              AI-powered resume tools
            </p>
          </div>
        </SidebarHeader>

        <SidebarContent>
          <SidebarMenu>
            {navItems.map((item) => (
              <SidebarMenuItem key={item.label}>
                <SidebarMenuButton asChild isActive={pathname === item.to}>
                  <NavLink to={item.to}>
                    <item.icon />
                    <span>{item.label}</span>
                  </NavLink>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarContent>
      </Sidebar>

      <SidebarInset>
        <main className="flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}
