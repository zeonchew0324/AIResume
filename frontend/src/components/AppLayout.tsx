import { NavLink, Outlet, useLocation } from "react-router-dom";
import {
  FileSearch,
  Wand2,
  ChevronLeft,
  ChevronRight,
  Book,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarInset,
  SidebarProvider,
  useSidebar,
} from "@/components/ui/sidebar";
import { StarfieldBackground } from "@/components/ui/starfield";

function SidebarCollapseButton() {
  const { state, toggleSidebar } = useSidebar();
  const collapsed = state === "collapsed";
  return (
    <button
      onClick={toggleSidebar}
      className="flex w-full items-center justify-center rounded-md p-2 text-sidebar-foreground/60 hover:bg-sidebar-accent hover:text-sidebar-foreground transition-colors"
    >
      {collapsed ? (
        <ChevronRight className="size-4" />
      ) : (
        <ChevronLeft className="size-4" />
      )}
    </button>
  );
}

const navItems = [
  { label: "Analyze Resume", icon: FileSearch, to: "/analyze" },
  { label: "Improve Resume", icon: Wand2, to: "/improve" },
  { label: "Cover Letter", icon: Book, to: "/cover-letter" },
];

export default function AppLayout() {
  const { pathname } = useLocation();

  return (
    <SidebarProvider className="h-svh">
      <Sidebar collapsible="icon">
        <SidebarHeader className="border-b border-sidebar-border">
          <div className="px-2 py-3">
            <h2 className="text-base font-semibold tracking-tight text-sidebar-foreground group-data-[state=collapsed]:hidden">
              AIResume
            </h2>
            <p className="text-xs text-sidebar-foreground/60 group-data-[state=collapsed]:hidden">
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

        <SidebarFooter className="border-t border-sidebar-border">
          <SidebarCollapseButton />
        </SidebarFooter>
      </Sidebar>

      <SidebarInset className="relative overflow-hidden">
        <StarfieldBackground className="!absolute" count={300} speed={0.3}>
          <main className="relative z-10 h-full overflow-auto p-6">
            <Outlet />
          </main>
        </StarfieldBackground>
      </SidebarInset>
    </SidebarProvider>
  );
}
