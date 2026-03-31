import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarProvider,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { FileSearch, Wand2, LayoutDashboard } from "lucide-react";

const navItems = [
  { label: "Overview", icon: LayoutDashboard, href: "#" },
  { label: "Analyze Resume", icon: FileSearch, href: "#analyze" },
  { label: "Improve Resume", icon: Wand2, href: "#improve" },
];

export default function Dashboard() {
  return (
    <SidebarProvider>
      <Sidebar>
        <SidebarHeader>
          <div className="px-2 py-3">
            <h2 className="text-base font-semibold tracking-tight text-sidebar-foreground">
              AIResume
            </h2>
            <p className="text-xs text-sidebar-foreground/60">Resume tools</p>
          </div>
        </SidebarHeader>

        <SidebarContent>
          <SidebarMenu>
            {navItems.map((item) => (
              <SidebarMenuItem key={item.label}>
                <SidebarMenuButton asChild tooltip={item.label}>
                  <a href={item.href}>
                    <item.icon />
                    <span>{item.label}</span>
                  </a>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarContent>

        <SidebarFooter>
          <div className="px-2 py-2 text-xs text-sidebar-foreground/40">
            v1.0.0
          </div>
        </SidebarFooter>
      </Sidebar>
    </SidebarProvider>
  );
}
